/*******************************************************
* matrixmult.cpp
 *
 * Multiplies two square matrices of size n x n.
Time  0.1160420 with kernel size 4x16, block size 4
 *******************************************************/

#include <iostream>
#include <vector>
#include <cassert>

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

const std::pair<int,int> kernel_size = {4, 16}; // second number should be a multiple of BLOCK_SIZE
const   int BLOCK_SIZE = 4;
const   int alignment = 64;
const   int VECTOR_SIZE = sizeof(int)*BLOCK_SIZE;

typedef int intvec __attribute__ ((vector_size(VECTOR_SIZE))); // 32 bytes = 8 integers (8 * 4 bytes/int)
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

typedef std::vector<intvec, AlignedVectorAllocator<intvec, alignment>> intvec_vec;


#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx512f")))
__attribute__((target("avx2")))
#else
#endif
inline void kernel(const intvec_vec &A, const intvec_vec &B, intvec_vec &C, int x, int y, int n, int NUM_BLOCKS_COL) { // x is row, y is *index of block*
    intvec ke[kernel_size.first][kernel_size.second/BLOCK_SIZE] = {};

    for (int k=0; k<n; k++) { // TODO: either loop unrolling, or some other way to iterate to make it faster (eg. reduce matrix index calculations)
        for (int i=x; i<x+kernel_size.first; ++i) {
            for (int j=y; j<y+kernel_size.second/BLOCK_SIZE; ++j) {
                intvec tmp = intvec{} + A[i*NUM_BLOCKS_COL + k/BLOCK_SIZE][k%BLOCK_SIZE]; // broadcast A
                ke[i-x][j-y] += tmp * B[k*NUM_BLOCKS_COL + j];
            }
        }
    }

    for (int i=x; i<x+kernel_size.first; i++) {
        for (int j=y; j<y+kernel_size.second/BLOCK_SIZE; ++j) {
            C[i*NUM_BLOCKS_COL + j] = ke[i-x][j-y];
        }
    }
}

#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx512f")))
#else
#endif
int main() {

    assert(kernel_size.second % BLOCK_SIZE==0);
    int n;
    std::ios::sync_with_stdio(0);
    std::cin.tie(0);
    std::cout.tie(0);
    std::cin >> n;

    const int NUM_BLOCKS_COL = (n+kernel_size.second-1)/kernel_size.second * (kernel_size.second/BLOCK_SIZE); // pack BLOCK_SIZE ints per vector
    const int NUM_BLOCKS_ROW = (n+kernel_size.first-1)/kernel_size.first * kernel_size.first; // pack BLOCK_SIZE ints per vector
//    std::cerr << "NUM_BLOCKS_COL: " << NUM_BLOCKS_COL << "\n";
    // Create matrices A, B, and C (all n x n, blocked)
    intvec_vec A(NUM_BLOCKS_ROW*NUM_BLOCKS_COL); // [[NUM_BLOCKS_COL][NUM_BLOCKS_COL] ... ]
    intvec_vec B(NUM_BLOCKS_ROW*NUM_BLOCKS_COL);
    intvec_vec C(NUM_BLOCKS_ROW*NUM_BLOCKS_COL, intvec{});


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
    for (int i=0; i<NUM_BLOCKS_ROW; i+=kernel_size.first) {
        for (int j=0; j<NUM_BLOCKS_COL; j+=kernel_size.second/BLOCK_SIZE) {
            kernel(A, B, C, i, j, n, NUM_BLOCKS_COL);
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
