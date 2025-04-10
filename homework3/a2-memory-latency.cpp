#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>

static inline uint64_t get_cycles() {
    uint64_t val;
    // Use CNTVCT_EL0 (Virtual Timer Count Register) on Apple Silicon
    asm volatile("mrs %0, CNTVCT_EL0" : "=r" (val));
    return val;
}

void memory_fence() {
    asm volatile("dmb ish" ::: "memory");
}

// Size larger than L1 cache
const size_t ARRAY_SIZE = 64 * 1024 * 1024; // 64MB
const int NUM_ITERATIONS = 1000000;
const int STRIDE_SIZE = 64; // Typical cache line size

inline void flush_cache_line(void* ptr) {
    // ARM64 cache line flush
    asm volatile("dc civac, %0" : : "r" (ptr) : "memory");
}

uint64_t measure_access_time(int* ptr) {
    uint64_t start, end;
    volatile int dummy;

    memory_fence();
    start = get_cycles();

    dummy = *ptr;

    memory_fence();
    end = get_cycles();

    return end - start;
}

int main() {
    // Allocate memory
    std::vector<int> array(ARRAY_SIZE);
    std::generate(array.begin(), array.end(), std::rand);

    uint64_t cached_total = 0;
    uint64_t uncached_total = 0;

    // Measure cached vs uncached access
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        int index = (rand() % (ARRAY_SIZE - STRIDE_SIZE)) & ~(STRIDE_SIZE - 1);
        int* ptr = &array[index];

        // First access (uncached)
        flush_cache_line(ptr);
        memory_fence();
        uncached_total += measure_access_time(ptr);

        // Second access (cached)
        cached_total += measure_access_time(ptr);
    }

    double avg_cached = static_cast<double>(cached_total) / NUM_ITERATIONS;
    double avg_uncached = static_cast<double>(uncached_total) / NUM_ITERATIONS;

    std::cout << "Average cached access latency: " << avg_cached << " cycles" << std::endl;
    std::cout << "Average uncached access latency: " << avg_uncached << " cycles" << std::endl;
    std::cout << "Difference (cache impact): " << (avg_uncached - avg_cached) << " cycles" << std::endl;

    return 0;
}
