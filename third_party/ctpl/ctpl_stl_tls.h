/*********************************************************
*
*  Copyright (C) 2014 by Vitaliy Vitsentiy
*  https://github.com/vit-vit/CTPL
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
*
*  October 2015, Jonathan Hadida:
*   - rename, reformat and comment some functions
*   - add a restart function
*  	- fix a few unsafe spots, eg:
*  		+ order of testing in setup_thread;
*  		+ atomic guards on pushes;
*  		+ make clear_queue private
*
*  August 2020, Renato Schmidt:
*   - Added support for thread local storage
*
*********************************************************/


#ifndef CTPL_H_INCLUDED
#define CTPL_H_INCLUDED

#if defined(__SANITIZE_ADDRESS__)
#include <sanitizer/lsan_interface.h>  // __lsan_disable: TLS tags are
                                       // deliberately never freed (below)
#endif

#include <functional>
#include <thread>
#include <atomic>
#include <vector>
#include <memory>
#include <future>
#include <mutex>
#include <queue>


// thread pool to run user's functors with signature
//      ret func(size_t id, other_params)
// where id is the index of the thread that runs the functor
// ret is some return type


namespace ctpl {

    namespace detail {
        template <typename T>
        class Queue {
        public:
            bool push(T const & value) {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_queue.push(value);
                return true;
            }
            // deletes the retrieved element, do not use for non integral types
            bool pop(T & v) {
                std::unique_lock<std::mutex> lock(m_mutex);
                if (m_queue.empty())
                    return false;
                v = m_queue.front();
                m_queue.pop();
                return true;
            }
            bool empty() {
                std::unique_lock<std::mutex> lock(m_mutex);
                return m_queue.empty();
            }
        private:
            std::queue<T> m_queue;
            std::mutex    m_mutex;
        };
    }

    template<class TLS>
    class thread_pool_tls {
    public:
        using base_func_type = std::function<void(size_t, TLS&)>;
        using init_func_type = std::function<void(size_t, std::shared_ptr<TLS>&)>;
        thread_pool_tls() { this->init(); }
        thread_pool_tls(size_t nThreads, init_func_type tlsInitFunction = nullptr) {
            this->init(); 
            m_tls_init_function = tlsInitFunction;
            m_nThreads = nThreads; 
            this->resize(nThreads); 
        }

        // the destructor waits for all the functions in the queue to be finished
        ~thread_pool_tls() { this->interrupt(false); }

        // get the number of running threads in the pool
        inline size_t size() const { return m_threads.size(); }

        // number of idle threads
        inline size_t n_idle() const { return ma_n_idle; }

        // get a specific thread
        inline std::thread & get_thread( size_t i ) { return *m_threads.at(i); }


        // restart the pool
        void restart()
        {
            this->interrupt(false); // finish all existing tasks but prevent new ones
            this->init(); // reset atomic flags
            this->resize(m_nThreads);
        }

        // change the number of threads in the pool
        // should be called from one thread, otherwise be careful to not interleave, also with this->interrupt()
        void resize(size_t nThreads)
        { if (!ma_kill && !ma_interrupt) {

            size_t oldNThreads = m_threads.size();
            if (oldNThreads <= nThreads) {  // if the number of threads is increased

                m_threads .resize(nThreads);
                m_abort   .resize(nThreads);
                m_tls     .resize(nThreads);

                for (size_t i = oldNThreads; i < nThreads; ++i)
                {
                    m_abort[i] = std::make_shared<std::atomic<bool>>(false);
                    this->setup_thread(i);
                }
            }
            else {  // the number of threads is decreased

                for (size_t i = oldNThreads - 1; i >= nThreads; --i)
                {
                    *m_abort[i] = true;  // this thread will finish
                    m_threads[i]->detach();
                }
                {
                    // stop the detached threads that were waiting
                    std::unique_lock<std::mutex> lock(m_mutex);
                    m_cond.notify_all();
                }
                m_threads .resize(nThreads); // safe to delete because the threads are detached
                m_abort   .resize(nThreads); // safe to delete because the threads have copies of shared_ptr of the flags, not originals
                m_tls     .resize(nThreads);
            }
        }}

        // wait for all computing threads to finish and stop all threads
        // may be called asynchronously to not pause the calling thread while waiting
        // if kill == true, all the functions in the queue are run, otherwise the queue is cleared without running the functions
        void interrupt( bool kill = false )
        {
            if (kill) {
                if (ma_kill) return;
                ma_kill = true;

                for (size_t i = 0, n = this->size(); i < n; ++i)
                    *m_abort[i] = true;  // command the threads to stop
            }
            else {
                if (ma_interrupt || ma_kill) return;
                ma_interrupt = true;  // give the waiting threads a command to finish
            }

            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_cond.notify_all();  // stop all waiting threads
            }
            // wait for the computing threads to finish
            for (size_t i = 0; i < m_threads.size(); ++i)
            {
                if (m_threads[i]->joinable())
                    m_threads[i]->join();
            }

            this->clear_queue();

            m_threads .clear();
            m_abort   .clear();
            m_tls     .clear();
        }

        template<typename F, typename... Rest>
        auto push(F && f, Rest&&... rest) ->std::future<decltype(f(std::declval<size_t>(), std::declval<TLS&>(), rest...))>
        {
            if (!ma_kill && !ma_interrupt)
            {
                auto pck = std::make_shared<std::packaged_task< decltype(f(std::declval<size_t>(), std::declval<TLS&>(), rest...)) (size_t, TLS&) >>(
                    std::bind(std::forward<F>(f), std::placeholders::_1, std::placeholders::_2, std::forward<Rest>(rest)...)
                );
                auto _f  = new base_func_type([pck](size_t id, TLS& tls){ (*pck)(id, tls); });

                m_queue.push(_f);
                std::unique_lock<std::mutex> lock(m_mutex);
                m_cond.notify_one();
                return pck->get_future();
            }
            else return std::future<decltype(f(std::declval<size_t>(), std::declval<TLS&>(), rest...))>();
        }

        // run the user's function that accepts argument size_t - id of the running thread. returned value is templatized
        // operator returns std::future, where the user can get the result and rethrow the catched exceptions
        template<typename F>
        auto push(F && f) ->std::future<decltype(f(std::declval<size_t>(), std::declval<TLS&>()))>
        {
            if (!ma_kill && !ma_interrupt)
            {
                auto pck = std::make_shared<std::packaged_task< decltype(f(std::declval<size_t>(), std::declval<TLS&>())) (size_t, TLS&) >>(std::forward<F>(f));
                auto _f  = new base_func_type([pck](size_t id, TLS& tls){ (*pck)(id, tls); });

                m_queue.push(_f);
                std::unique_lock<std::mutex> lock(m_mutex);
                m_cond.notify_one();
                return pck->get_future();
            }
            else return std::future<decltype(f(std::declval<size_t>(), std::declval<TLS&>()))>();
        }


        /// Read-only access to per-thread TLS slots. Caller must ensure the
        /// pool stays alive while iterating. Each slot is a shared_ptr that
        /// is non-null once the worker thread has run its init function; the
        /// init runs lazily on the worker's first task, so callers should
        /// only iterate after at least one task has run on each worker.
        const std::vector<std::shared_ptr<TLS>>& tls_slots() const {
            return m_tls;
        }


    private:


        // deleted
        thread_pool_tls(const thread_pool_tls &);// = delete;
        thread_pool_tls(thread_pool_tls &&);// = delete;
        thread_pool_tls & operator=(const thread_pool_tls &);// = delete;
        thread_pool_tls & operator=(thread_pool_tls &&);// = delete;

        // clear all tasks
        void clear_queue()
        {
            base_func_type * _f = nullptr;
            while (m_queue.pop(_f))
                delete _f; // empty the queue
        }

        // reset all flags
        void init() { ma_n_idle = 0; ma_kill = false; ma_interrupt = false; }

        // each thread pops jobs from the queue until:
        //  - the queue is empty, then it waits (idle)
        //  - its abort flag is set (terminate without emptying the queue)
        //  - a global interrupt is set, then only idle threads terminate
        void setup_thread( size_t i )
        {
            // a copy of the shared ptr to the abort
            std::shared_ptr<std::atomic<bool>> abort_ptr(m_abort[i]);

            // pointer to the shared ptr to the thread's tls
            std::shared_ptr<TLS>* tls_sharedptr_ptr = &(m_tls[i]);

            init_func_type tls_init_function = m_tls_init_function;

            auto f = [this, i, abort_ptr, tls_sharedptr_ptr, tls_init_function]()
            {
                std::atomic<bool> & abort = *abort_ptr;
                
                std::shared_ptr<TLS>& tls_ptr_ref = *tls_sharedptr_ptr;
                if (tls_init_function) {
                    tls_init_function(i, tls_ptr_ref);
                }
                // If no init function was provided, default-construct the TLS
                // so the dereference below is never UB on a null shared_ptr.
                if (!tls_ptr_ref) {
                    // Pool threads park forever (the pool is deliberately
                    // never destroyed — see scan_pool.cpp), so this
                    // allocation is never released. Mark it as an
                    // intentional leak or LeakSanitizer fails the test
                    // binary at exit.
#if defined(__SANITIZE_ADDRESS__)
                    __lsan_disable();
#endif
                    tls_ptr_ref = std::make_shared<TLS>();
#if defined(__SANITIZE_ADDRESS__)
                    __lsan_enable();
#endif
                }                TLS& tls = *tls_ptr_ref;

                base_func_type * _f = nullptr;
                bool more_tasks = m_queue.pop(_f);

                while (true)
                {
                    while (more_tasks) // if there is anything in the queue
                    {
                        // at return, delete the function even if an exception occurred
                        std::unique_ptr<base_func_type> func(_f);
                        (*_f)(i, tls);
                        if (abort)
                            return; // return even if the queue is not empty yet
                        else
                            more_tasks = m_queue.pop(_f);
                    }

                    // the queue is empty here, wait for the next command
                    std::unique_lock<std::mutex> lock(m_mutex);
                    ++ma_n_idle;
                    m_cond.wait(lock, [this, &_f, &more_tasks, &abort](){ more_tasks = m_queue.pop(_f); return abort || ma_interrupt || more_tasks; });
                    --ma_n_idle;
                    if ( ! more_tasks) return; // we stopped waiting either because of interruption or abort
                }
            };

            m_threads[i].reset( new std::thread(f) );
        }

        // ----------  =====  ----------

        std::vector<std::unique_ptr<std::thread>>        m_threads;
        std::vector<std::shared_ptr<std::atomic<bool>>>  m_abort;
        detail::Queue<base_func_type *>  m_queue;

        std::vector<std::shared_ptr<TLS>> m_tls;
        init_func_type m_tls_init_function;

        std::atomic<bool>    ma_interrupt, ma_kill;
        std::atomic<size_t>  ma_n_idle;
        std::atomic<size_t> m_nThreads;

        std::mutex               m_mutex;
        std::condition_variable  m_cond;
    };
}

#endif
