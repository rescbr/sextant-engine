// I8MM precision spike: does scaling Lloyd-Max levels to int8 lose recall?
// Measures recall@10 for float ADC vs int8 ADC on Cohere 100k.
//
// Usage: ./spike_i8mm_prec <base.fbin> <query.fbin>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <set>

struct Fbin { uint32_t n, dim; std::vector<float> data; };
Fbin read_fbin(const char* path) {
    FILE* f = fopen(path, "rb"); if (!f) exit(1);
    Fbin fb; fread(&fb.n, 4, 1, f); fread(&fb.dim, 4, 1, f);
    fb.data.resize((size_t)fb.n * fb.dim);
    fread(fb.data.data(), 4, fb.data.size(), f); fclose(f); return fb;
}

std::vector<float> lloyd_max_1d(std::vector<float> data, uint32_t K) {
    std::sort(data.begin(), data.end());
    std::vector<float> levels(K);
    for (uint32_t k = 0; k < K; ++k)
        levels[k] = data[static_cast<size_t>(static_cast<double>(k)/(K-1)*(data.size()-1))];
    std::sort(levels.begin(), levels.end());
    for (uint32_t it = 0; it < 30; ++it) {
        std::vector<float> b(K-1);
        for (uint32_t i = 0; i < K-1; ++i) b[i] = 0.5f*(levels[i]+levels[i+1]);
        std::vector<double> sums(K,0); std::vector<uint64_t> cnt(K,0);
        for (float v : data) {
            uint32_t idx = std::upper_bound(b.begin(),b.end(),v)-b.begin();
            sums[idx]+=v; cnt[idx]++;
        }
        bool c=true;
        for (uint32_t k=0;k<K;++k) if(cnt[k]>0){float n=sums[k]/cnt[k];if(fabsf(n-levels[k])>1e-6f)c=false;levels[k]=n;}
        if(c) break;
    }
    std::sort(levels.begin(), levels.end());
    return levels;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <base.fbin> <query.fbin>\n", argv[0]); return 1; }
    auto fb = read_fbin(argv[1]);
    auto fq = read_fbin(argv[2]);
    uint32_t n = fb.n, dim = fb.dim;
    const uint32_t K = 16;

    // Train on 5k sample
    std::mt19937 rng(42);
    std::vector<uint32_t> idx(n); std::iota(idx.begin(),idx.end(),0);
    std::shuffle(idx.begin(),idx.end(),rng);
    uint32_t tn = std::min(5000u, n);
    std::vector<float> sample((size_t)tn*dim);
    for (uint32_t i=0;i<tn;++i) memcpy(&sample[(size_t)i*dim],&fb.data[(size_t)idx[i]*dim],dim*4);

    // Per-dim Lloyd-Max levels
    std::vector<float> levels((size_t)dim*K);
    std::vector<float> bounds((size_t)dim*(K-1));
    for (uint32_t d=0;d<dim;++d) {
        std::vector<float> col(tn);
        for (uint32_t i=0;i<tn;++i) col[i]=sample[(size_t)i*dim+d];
        auto lv = lloyd_max_1d(col, K);
        memcpy(&levels[(size_t)d*K], lv.data(), K*4);
        for (uint32_t i=0;i<K-1;++i) bounds[(size_t)d*(K-1)+i]=0.5f*(lv[i]+lv[i+1]);
    }

    // Encode all vectors
    uint32_t cb = dim/2;
    std::vector<uint8_t> codes((size_t)n*cb);
    for (uint32_t i=0;i<n;++i) {
        const float* v=&fb.data[(size_t)i*dim];
        uint8_t* c=&codes[(size_t)i*cb];
        for (uint32_t d=0;d<dim;d+=2) {
            auto i0=std::upper_bound(&bounds[(size_t)d*(K-1)],&bounds[(size_t)d*(K-1)]+K-1,v[d])-&bounds[(size_t)d*(K-1)];
            uint32_t i1=0;
            if(d+1<dim) i0?0:0; // suppress unused warning
            if(d+1<dim) i1=std::upper_bound(&bounds[(size_t)(d+1)*(K-1)],&bounds[(size_t)(d+1)*(K-1)]+K-1,v[d+1])-&bounds[(size_t)(d+1)*(K-1)];
            c[d/2]=(i0&0xF)|((i1&0xF)<<4);
        }
    }

    // Decode to int8 (global scale)
    float lmax=0; for(float v:levels) lmax=std::max(lmax,fabsf(v));
    float lscale=127.0f/lmax;
    std::vector<int8_t> levels_i8((size_t)dim*K);
    for(size_t i=0;i<levels.size();++i) levels_i8[i]=static_cast<int8_t>(roundf(levels[i]*lscale));

    // Decode all vectors to int8
    std::vector<int8_t> dec_i8((size_t)n*dim);
    for (uint32_t i=0;i<n;++i) {
        const uint8_t* c=&codes[(size_t)i*cb];
        for (uint32_t d=0;d<dim;d+=2) {
            uint8_t b=c[d/2];
            dec_i8[(size_t)i*dim+d]=levels_i8[(size_t)d*K+(b&0xF)];
            if(d+1<dim) dec_i8[(size_t)i*dim+d+1]=levels_i8[(size_t)(d+1)*K+((b>>4)&0xF)];
        }
    }

    // Quantize queries to int8
    float qmax=0; for(float v:fq.data) qmax=std::max(qmax,fabsf(v));
    float qscale=127.0f/qmax;
    std::vector<int8_t> q_i8((size_t)fq.n*dim);
    for(size_t i=0;i<fq.data.size();++i) q_i8[i]=static_cast<int8_t>(roundf(fq.data[i]*qscale));

    // Measure recall@10 for Q=200
    uint32_t Q=std::min(200u,fq.n);
    uint32_t cf=0, ci=0;
    float total_float_err=0, total_i8mm_err=0;

    for (uint32_t qi=0; qi<Q; ++qi) {
        const float* query=&fq.data[(size_t)qi*dim];
        const int8_t* qi8=&q_i8[(size_t)qi*dim];

        std::vector<std::pair<float,uint32_t>> df(n), di(n), exact(n);
        for (uint32_t i=0;i<n;++i) {
            const uint8_t* c=&codes[(size_t)i*cb];
            const int8_t* dv=&dec_i8[(size_t)i*dim];

            // Float ADC (IP: higher = closer)
            float fd=0;
            for (uint32_t d=0;d<dim;d+=2) {
                uint8_t b=c[d/2];
                fd+=query[d]*levels[(size_t)d*K+(b&0xF)];
                if(d+1<dim) fd+=query[d+1]*levels[(size_t)(d+1)*K+((b>>4)&0xF)];
            }
            df[i]={fd,i};

            // I8MM ADC (int32, higher = closer after scaling)
            int64_t id=0;
            for (uint32_t d=0;d<dim;++d) id+=(int64_t)qi8[d]*dv[d];
            di[i]={(float)id,i};

            // Exact float dot
            float ed=0;
            for (uint32_t d=0;d<dim;++d) ed+=query[d]*fb.data[(size_t)i*dim+d];
            exact[i]={ed,i};
        }

        std::nth_element(exact.begin(),exact.begin()+10,exact.end(),[](auto&a,auto&b){return a.first>b.first;});
        std::nth_element(df.begin(),df.begin()+10,df.end(),[](auto&a,auto&b){return a.first>b.first;});
        std::nth_element(di.begin(),di.begin()+10,di.end(),[](auto&a,auto&b){return a.first>b.first;});

        std::set<uint32_t> gt,f8,i8;
        for(uint32_t k=0;k<10;++k){gt.insert(exact[k].second);f8.insert(df[k].second);i8.insert(di[k].second);}
        for(uint32_t id:gt){if(f8.count(id))cf++; if(i8.count(id))ci++;}

        if(qi==0) for(uint32_t i=0;i<500;++i) {
            total_float_err+=fabsf(df[i].first-exact[i].first);
            total_i8mm_err+=fabsf(di[i].first/(qscale*lscale)-exact[i].first);
        }
    }

    printf("Float ADC recall@10: %.1f%%\n", 100.0f*cf/(Q*10));
    printf("I8MM  ADC recall@10: %.1f%%\n", 100.0f*ci/(Q*10));
    printf("Float sim err (Q0,500): %.6f\n", total_float_err/500);
    printf("I8MM  sim err (Q0,500): %.6f\n", total_i8mm_err/500);
    printf("Level scale: %.1f (max=%.4f), Query scale: %.1f (max=%.4f)\n",
           lscale,lmax,qscale,qmax);
    return 0;
}
