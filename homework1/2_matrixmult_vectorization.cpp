/*******************************************************
* matrixmult.cpp
 *
 * Multiplies two square matrices of size n x n.
Time  0.2651s
 *******************************************************/

#include <iostream>
#include <vector>

#include <omp.h>

// measure the amount of clock cycles used. Useful only on x86_64 (notably not on ARM or x86)
typedef unsigned long long ull;

static inline uint64_t rdtsc() {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int lo, hi;
    __asm__ __volatile__("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
#else
    return 0;
#endif
}
const   int BLOCK_SIZE = 8;
const   int VECTOR_SIZE = sizeof(int)*BLOCK_SIZE;
typedef int intvec __attribute__ ((vector_size(VECTOR_SIZE))); // 32 bytes = 8 integers (8 * 4 bytes/int)
const   int alignment = 64;
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


#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx512f")))
#else
#endif
int main() {
    int n;
    std::ios::sync_with_stdio(0);
    std::cin.tie(0);
    std::cout.tie(0);
    std::cin >> n;

    int NUM_BLOCKS_COL = (n+BLOCK_SIZE-1)/BLOCK_SIZE; // pack BLOCK_SIZE ints per vector
    // Create matrices A, B, and C (all n x n, blocked)
    std::vector<intvec, AlignedVectorAllocator<intvec, alignment>> A(n * NUM_BLOCKS_COL);
    std::vector<intvec, AlignedVectorAllocator<intvec, alignment>> B(n * NUM_BLOCKS_COL);
    std::vector<intvec, AlignedVectorAllocator<intvec, alignment>> C(n * NUM_BLOCKS_COL, intvec{});


    // Read matrix A
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cin >> A[i*NUM_BLOCKS_COL + j/BLOCK_SIZE][j%BLOCK_SIZE];
        }
    }

    // Read matrix B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cin >> B[i*NUM_BLOCKS_COL + j/BLOCK_SIZE][j%BLOCK_SIZE];
        }
    }

    // TODO: perform matrix multiplication A x B and write into C: C = A x B
    // YOUR CODE HERE
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j+=BLOCK_SIZE) {
            intvec accum = {};
            for (int k=0; k<n; k++) {
                intvec v1 = intvec{} + A[i * NUM_BLOCKS_COL + k / BLOCK_SIZE][k % BLOCK_SIZE];
                intvec w1 = B[k * NUM_BLOCKS_COL + j / BLOCK_SIZE];
                accum += v1 * w1;
            }
            C[i*NUM_BLOCKS_COL + j/BLOCK_SIZE] = accum;
        }
    }

    std::cout << "The resulting matrix C = A x B is:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << C[i*NUM_BLOCKS_COL+j/BLOCK_SIZE][j%BLOCK_SIZE] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
