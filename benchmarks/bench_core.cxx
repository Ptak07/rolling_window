#include <benchmark/benchmark.h>
#include "MultisetMedian.hpp"
#include <vector>
#include <random>
#include <cstddef>

std::vector<double> generate_market_data(std::size_t size) {
    std::vector<double> data(size);
    std::mt19937 gen(42); 
    std::normal_distribution<double> dist(100.0, 5.0);
    
    for (std::size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
    return data;
}

const auto MARKET_DATA = generate_market_data(100000);

template <class MedianAlgo>
static void BM_RollingMedian(benchmark::State& state) {
    std::size_t window_size = static_cast<std::size_t>(state.range(0));
    
    for (auto _ : state) {
        MedianAlgo engine(window_size);
        
        for (double price : MARKET_DATA) {
            engine.update(price);
            benchmark::DoNotOptimize(engine.get_median()); 
        }
    }
}

BENCHMARK_TEMPLATE(BM_RollingMedian, MultisetMedian)->Arg(10)->Arg(100)->Arg(1000);