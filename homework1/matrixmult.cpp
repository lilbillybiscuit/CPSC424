/*******************************************************
 * matrixmult.cpp
 *
 * Multiplies two square matrices of size n x n.
 *******************************************************/

#include <iostream>
#include <vector>

#include <omp.h>

#include <immintrin.h>

#include <cstdlib>

// measure the amount of clock cycles used. Useful only on x86_64 (notably not on ARM or x86)
typedef unsigned long long ull;
static inline ull rdtsc() {
    unsigned int lo, hi;
    __asm__ __volatile__("rdtsc" : "=a" (lo), "=d" (hi));
    return ((ull)hi << 32) | lo;
}

#define ALIGNMENT 256

template <class T, std::size_t alignment>
struct AlignedVectorAllocator
{

    typedef T value_type;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;

    AlignedVectorAllocator() noexcept = default;

    template <typename U>
    struct rebind {
        typedef AlignedVectorAllocator<U, alignment> other;
    };


    template<class U> constexpr AlignedVectorAllocator(const AlignedVectorAllocator<U, alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n)
    {
        void *ptr = std::aligned_alloc(alignment, n*sizeof (T));
        if (ptr == nullptr)
        {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
        free(p);
    }
};


int main() {
    int n;
    std::ios::sync_with_stdio(0);
    std::cin.tie(0);
    std::cout.tie(0);

    std::cin >> n;

    int roundedn = (n+8-1)/8*8; // pack 8 ints per vector


    // Create matrices A, B, and C (all n x n)
    std::vector<std::vector<int, AlignedVectorAllocator<int, 32>>> A(n, std::vector<int, AlignedVectorAllocator<int, 32>>(roundedn));
    std::vector<std::vector<int, AlignedVectorAllocator<int, 32>>> B(n, std::vector<int, AlignedVectorAllocator<int, 32>>(roundedn));
    std::vector<std::vector<int, AlignedVectorAllocator<int, 32>>> C(n, std::vector<int, AlignedVectorAllocator<int, 32>>(roundedn, 0));

    // Read matrix A
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cin >> A[i][j];
        }
    }

    // Read matrix B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cin >> B[i][j]; // NOTE: that this is transposed
        }
    }

    // TODO: perform matrix multiplication A x B and write into C: C = A x B
    // YOUR CODE HERE
    ull start = rdtsc();
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j+=8) { // 4bytes*16 = 64Bsize of cache line. TODO: CHANGE TO 16 to try
            __m256i accum = _mm256_setzero_si256();
            for (int k=0; k<n; k++) {
                __m256i v = _mm256_set1_epi32(A[i][k]);
                __m256i w = _mm256_load_si256((__m256i*)&B[k][j]);
                __m256i res = _mm256_mullo_epi32(v,w);
                accum = _mm256_add_epi32(accum, res);
            }
            _mm256_store_si256((__m256i*)&C[i][j], accum);
        }
    }
    ull end = rdtsc();
    std::cout << "Used " << end-start << " cycles\n";

    std::cout << "The resulting matrix C = A x B is:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << C[i][j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
