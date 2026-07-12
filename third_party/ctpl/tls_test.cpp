// Build with
// g++ -g -o tls_test -std=c++17 -lpthread -I.. tls_test.cpp

#include "ctpl/ctpl_stl_tls.h"
#include <iostream>
#include <string>
#include <random>

class MyTLS {
public:
    std::string wololo;
    int num = 42;
    int ctornum = 0;
    MyTLS(int x): ctornum(x), wololo("asdf") {};
};

std::mutex cout_mut;

void funfunfun(size_t id, MyTLS& tls) {
    { //Critical Section
        std::lock_guard<std::mutex> lock{cout_mut};
        std::cout << "Hello from TLS id " << std::to_string(tls.ctornum) << std::endl;
    }
    
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    { //Critical Section
        std::lock_guard<std::mutex> lock{cout_mut};
        std::cout << "Bye from TLS id " << std::to_string(tls.ctornum) << std::endl;
    }
}

void notfun(size_t id) {};

int main(int argc, char **argv) {

    ctpl::thread_pool_tls<MyTLS> pool(8, [](size_t id, std::shared_ptr<MyTLS>& tls){
        //auto rnd = std::random_device();
        //int r = (int)(rnd());
        tls.reset(new MyTLS(id));
    });

    for(size_t i = 0; i < 32; i++){
        pool.push(funfunfun);
        //pool.push([](size_t id, MyTLS& tls){});
    }
}

