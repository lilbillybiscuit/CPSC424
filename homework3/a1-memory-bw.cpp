#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <chrono>
#include <omp.h>

// Use larger blocks for memory access
constexpr size_t BLOCK_SIZE = 16384; // 16KB blocks like sysbench
constexpr size_t BUFFER_SIZE = 1ULL << 30; // 1GB
constexpr size_t ALIGNMENT = 64;

// NUMA-aware allocation if available
void* allocate_aligned_memory(size_t size, size_t alignment) {
    void* ptr = nullptr;
#ifdef _GNU_SOURCE
    ptr = aligned_alloc(alignment, size);
#else
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
#endif
    return ptr;
}

int main() {
    char* buffer = static_cast<char*>(allocate_aligned_memory(BUFFER_SIZE, ALIGNMENT));
    if (!buffer) {
        std::cerr << "Memory allocation failed!\n";
        return 1;
    }

    memset(buffer, 1, BUFFER_SIZE);

    volatile uint64_t dummy_sum = 0;
    double total_bytes = 0;
    const int num_threads = omp_get_max_threads();

    // Calculate chunk size that's a multiple of block size
    const size_t chunk_size = (BUFFER_SIZE / num_threads) & ~(BLOCK_SIZE - 1);

    const auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel reduction(+:total_bytes)
    {
        const int tid = omp_get_thread_num();
        const char* ptr = buffer + tid * chunk_size;
        uint64_t local_sum = 0;
        size_t iterations = 0;

        // Temporary buffer aligned to cache line
        alignas(ALIGNMENT) char temp_buffer[BLOCK_SIZE];

        // Run for 5 seconds
        while (std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count() < 5.0) {
            for (size_t offset = 0; offset < chunk_size; offset += BLOCK_SIZE) {
                // Prefetch next block
                __builtin_prefetch(ptr + offset + BLOCK_SIZE, 0, 3);

                // Copy block to temporary buffer (this is what sysbench does)
                memcpy(temp_buffer, ptr + offset, BLOCK_SIZE);

                // Sum the block to prevent optimization
                for (size_t i = 0; i < BLOCK_SIZE; i += 8) {
                    local_sum += *reinterpret_cast<const uint64_t*>(temp_buffer + i);
                }
            }
            iterations++;
        }
        total_bytes += iterations * chunk_size;
        dummy_sum += local_sum;
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const double duration = std::chrono::duration<double>(end - start).count();
    const double bandwidth = total_bytes / (duration * 1e9);

    std::cout << "Measured bandwidth: " << bandwidth << " GB/s\n";
    std::cout << "Dummy checksum: " << dummy_sum << "\n";

    free(buffer);
    return 0;
}
