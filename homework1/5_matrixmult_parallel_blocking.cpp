/*******************************************************
* matrixmult.cpp
 *
 * Multiplies two square matrices of size n x n.
Time  0.1160420 with kernel size 4x16, block size 4
 *******************************************************/

#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <cstdint>

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

//const   int L3_CACHE_SIZE = 1024*1024*2; // 32 MB
const   int L2_CACHE_SIZE = 1024*1024; // 256 KB
const   int L1_CACHE_SIZE = 128*1024; // 32 KB

const int N_KERNEL_COLS_PER_L3 = 16; // for 2048 matrix, kernel width =32,  each column selection block takes 2048*4 = 8KB.

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


/*
 * Kernel function
 * updates C[x:x+kernel_size.first][y:y+kernel_size.second/BLOCK_SIZE]
 * multiply A[x:x+kernel_size.first][l:r] with B[l:r][y:y+kernel_size.second/BLOCK_SIZE]
 * x and y determine the continuous kernel_size block that is modified in C. They do not have to be
 */

#if defined(__x86_64__) || defined(_M_X64)

__attribute__((target("avx512f")))
__attribute__((target("avx2")))
#else
#endif
inline void kernel(const intvec_vec &A, const intvec_vec &B, intvec_vec &C,
                   int x, int y, // starting index of the BLOCK
                   int l, int r, // range for k (horizontal portion of row in A, vertical portion of column in B)
                   int stride) {
    intvec ke[kernel_size.first][kernel_size.second/BLOCK_SIZE] = {};

    for (int k=l; k<r; k++) { // TODO: either loop unrolling, or some other way to iterate to make it faster (eg. reduce matrix index calculations)
                              // TODO: potentially support dynamic kernel sizes at the cost of not having constant loop sizes (loop unenrolling)
        for (int i=x; i<x+kernel_size.first; ++i) {
            for (int j=y; j<y+kernel_size.second/BLOCK_SIZE; ++j) {
                intvec tmp = intvec{} + A[i*stride + k/BLOCK_SIZE][k%BLOCK_SIZE]; // broadcast A
                ke[i-x][j-y] += tmp * B[k*stride + j];
            }
        }
    }

    for (int i=x; i<x+kernel_size.first; i++) {
        for (int j=y; j<y+kernel_size.second/BLOCK_SIZE; ++j) {
            C[i*stride + j] += ke[i-x][j-y];
        }
    }
}

#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx512f")))
#else
#endif
int main() {
    assert(kernel_size.second % BLOCK_SIZE==0);
    assert(N_KERNEL_COLS_PER_L3 > 0);
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
//    for (int i=0; i<NUM_BLOCKS_ROW; i+=kernel_size.first) {
//        for (int j=0; j<NUM_BLOCKS_COL; j+=kernel_size.second/BLOCK_SIZE) {
//            kernel(A, B, C, i, j, 0, n, NUM_BLOCKS_COL);
//        }
//    }

    std::cerr << "N_KERNEL_COLS_PER_L3: " << N_KERNEL_COLS_PER_L3*kernel_size.second << "\n";
    const int N_KERNEL_ROWS_PER_L2 = (L2_CACHE_SIZE / (sizeof(int) * NUM_BLOCKS_COL * BLOCK_SIZE)) / kernel_size.first; assert(N_KERNEL_ROWS_PER_L2 > 0); // for 2048 matrix, block size=4, each row selection block takes 2048*4 = 8KB.
    std::cerr << "N_KERNEL_ROWS_PER_L2: " << N_KERNEL_ROWS_PER_L2*kernel_size.first << "\n";
    const int N_KERNEL_ROWS_PER_L1 = L1_CACHE_SIZE / (N_KERNEL_COLS_PER_L3 * sizeof(int) * kernel_size.second) / kernel_size.first; assert(N_KERNEL_ROWS_PER_L1 > 0); // for 2048 matrix, block size=4, for a single column selection (block_size columns), we need block_size * block_size * 4 = 64B of data
    std::cerr << "N_KERNEL_ROWS_PER_L1: " << N_KERNEL_ROWS_PER_L1*kernel_size.first << "\n";

    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for collapse(2)
    for (int i3=0; i3<NUM_BLOCKS_COL; i3+=N_KERNEL_COLS_PER_L3 * kernel_size.second / BLOCK_SIZE) { // select columns of B (of size 4*n) to go into the L3 cache
        for (int i2=0; i2<NUM_BLOCKS_ROW; i2+= N_KERNEL_ROWS_PER_L2 * kernel_size.first) { // select rows of A (of size 4*n) to go into the L2 cache
            for (int i1=0; i1<NUM_BLOCKS_ROW; i1+= N_KERNEL_ROWS_PER_L1 * kernel_size.first) { // select rows of B (of size 4*n) to go into the L1d cache

                // here we are only considering processing whole kernels, the dimensions should be multiples of kernel_size.
                // if the block is not complete, then the std::min will make sure we don't go out of bounds
                int i1_end = i1 + N_KERNEL_ROWS_PER_L1 * kernel_size.first;
                int i2_end = i2 + N_KERNEL_ROWS_PER_L2 * kernel_size.first;
                int i3_end = i3 + N_KERNEL_COLS_PER_L3 * kernel_size.second / BLOCK_SIZE;

                int counter = 0;
                for (int x=i2; x<std::min(i2_end, NUM_BLOCKS_ROW); x+=kernel_size.first) { // select vertical kernel blocks of A
                    for (int y=i3; y<std::min(i3_end, NUM_BLOCKS_COL); y+= kernel_size.second / BLOCK_SIZE) { // select horizontal kernel blocks of B
                        counter++;
//                        std::cerr << "i1: " << i1 << " ending_range: " << ending_range << "\n";
                        kernel(A, B, C, x, y, i1, std::min(i1_end, n), NUM_BLOCKS_COL);
                    }
                }
//                std::cerr << "counter: " << counter << "\n";
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cerr << "Time in seconds: " << duration.count() / 1000000.0 << "\n";

    std::cout << "The resulting matrix C = A x B is:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << C[i*NUM_BLOCKS_COL+j/BLOCK_SIZE][j%BLOCK_SIZE] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
