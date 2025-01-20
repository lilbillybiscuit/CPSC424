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

    const int BLOCK_SIZE = 16;
    int roundedn = (n+BLOCK_SIZE-1)/BLOCK_SIZE*BLOCK_SIZE; // pack 8 ints per vector
    const int alignment = 64;


    // Create matrices A, B, and C (all n x n)
    std::vector<int, AlignedVectorAllocator<int, alignment>> A(n*roundedn);
    std::vector<int, AlignedVectorAllocator<int, alignment>> B(n*roundedn);
    std::vector<int, AlignedVectorAllocator<int, alignment>> C(n*roundedn, 0);

    // Read matrix A
    for (int i = 0; i < n; ++i) {
        int baseind = i*roundedn;
        for (int j = 0; j < n; ++j) {
            std::cin >> A[baseind+j];
        }
    }

    // Read matrix B
    for (int i = 0; i < n; ++i) {
        int baseind = i*roundedn;
        for (int j = 0; j < n; ++j) {
            std::cin >> B[baseind+j]; // NOTE: that this is transposed
        }
    }

    // TODO: perform matrix multiplication A x B and write into C: C = A x B
    // YOUR CODE HERE
    ull start = rdtsc();
    for (int i=0; i<n; i++) {
        int baseindi = i*roundedn;
        for (int j=0; j<n; j+=16) { // 4bytes*16 = 64Bsize of cache line. TODO: CHANGE TO 16 to try
            __m512i accum = _mm512_setzero_si512();

            for (int k=0; k<n; k++) {
                __m512i v1 = _mm512_set1_epi32(A[baseindi+k]);
                __m512i w1 = _mm512_load_si512((__m512i*)&B[k*roundedn + j]);
                __m512i res1 = _mm512_mullo_epi32(v1,w1);
                accum = _mm512_add_epi32(accum, res1);
            }
            _mm512_store_epi32((__m512i*)&C[baseindi + j], accum);
        }
    }
    ull end = rdtsc();
    std::cerr << "Used " << end-start << " cycles\n";

    std::cout << "The resulting matrix C = A x B is:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << C[i*roundedn + j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
