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
#include <cstdlib>

#include <omp.h>
#include <cstring>





const std::pair<int,int> kernel_size = {6, 32}; // second number should be a multiple of BLOCK_SIZE
const   int BLOCK_SIZE = 16;
const   int alignment = 64;
const   int VECTOR_SIZE = sizeof(int)*BLOCK_SIZE;

const   int CHAR_SIMD_STRIDE = 64; // on 512-bit registers, 512 bits = 64 bytes = 64 chars

//const   int L3_CACHE_SIZE = 1024*1024*2; // 32 MB
const   int L2_CACHE_SIZE = 1024*1024; // 256 KB
const   int L1_CACHE_SIZE = 128*1024; // 32 KB

const int N_KERNEL_COLS_PER_L3 = 48; // for 2048 matrix, kernel width =32,  each column selection block takes 2048*4 = 8KB.

typedef int intvec __attribute__ ((vector_size(VECTOR_SIZE))); // 32 bytes = 8 integers (8 * 4 bytes/int)
typedef char charvec __attribute__ ((vector_size(CHAR_SIMD_STRIDE))); // 64 bytes = 64 chars (64 * 1 byte/char)
#define SHOW_TIMER true
#define START_TIMER if (SHOW_TIMER) { \
        start = std::chrono::high_resolution_clock::now(); \
    }
#define END_TIMER(label) \
    if (SHOW_TIMER) { \
        auto end = std::chrono::high_resolution_clock::now(); \
        std::chrono::duration<double> elapsed = end - start; \
        std::cerr << "[" << label << "] time elapsed " << elapsed.count() << "s\n"; \
    }


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

static __attribute__((always_inline)) inline intvec *intvec_alloc(std::size_t n_bytes) {
    intvec* res =  static_cast<intvec*>(std::aligned_alloc(alignment, n_bytes*sizeof(intvec)));
    if (res == nullptr) {
        throw std::bad_alloc();
    }
    memset(res, 0, n_bytes*sizeof(intvec));
    return res;
}

/*
 * Kernel function
 * updates C[x:x+kernel_size.first][y:y+kernel_size.second/BLOCK_SIZE]
 * multiply A[x:x+kernel_size.first][l:r] with B[l:r][y:y+kernel_size.second/BLOCK_SIZE]
 * x and y determine the continuous kernel_size block that is modified in C. They do not have to be
 */

__attribute__((hot))
#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx512f")))
__attribute__((target("avx2")))
#else
#endif
inline __attribute__((always_inline)) void kernel(const intvec *__restrict__ A, const intvec * __restrict__ B, intvec *__restrict__ C,
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
__attribute__((hot))
__attribute__((target("avx512f")))
__attribute__((target("avx2")))
#else
#endif
inline __attribute__((always_inline)) void kernel_v2(const intvec *__restrict__ A, const intvec * __restrict__ B, intvec *__restrict__ C,
                   int x, int y, // starting index of the BLOCK
                   int l, int r, // range for k (horizontal portion of row in A, vertical portion of column in B)
                   int stride) {
    int rowStarts[kernel_size.first];
    for (int i=0; i<kernel_size.first; i++) {
        rowStarts[i] = (i+x)*stride;
    }

#define allVec(funcName) \
    funcName(0,0); funcName(1,0); funcName(2,0); funcName(3,0); funcName(4,0); funcName(5,0); \
    funcName(0,1); funcName(1,1); funcName(2,1); funcName(3,1); funcName(4,1); funcName(5,1);
    #define initvec(i,j) intvec v##i##j = intvec{} + 0;
    allVec(initvec)

    // kernel_size.first is now 6
    // kernel_size.second is now 32 -> 2 blocks HARDCODED
    #define broadcasted(i,j) intvec init##i##j = intvec{} + A[rowStarts[i] + k_over][k_mod]; \
        v##i##j += init##i##j * bVec##j;
    for (int k=l; k<r; k++) {
        int bIndex0 = (k*stride + y);
        int bIndex1 = (k*stride + y+1);
        intvec bVec0 = B[bIndex0];
        intvec bVec1 = B[bIndex1];

        int k_over = k/BLOCK_SIZE; int k_mod = k%BLOCK_SIZE;

        allVec(broadcasted)
    }

    #define addToC(i,j) C[rowStarts[i] + y+j] += v##i##j;
    allVec(addToC)

}


static unsigned char* buffer = nullptr;
static int input_index = 0, input_len = 0, buffer_len = 0;
const unsigned char SPACE = ' '-'0', NEWLINE = '\n' - '0', NEGATIVE = '-' - '0';

static __attribute__((always_inline)) inline void fastReadInput(intvec *__restrict__ A, intvec*__restrict__ B, int n) {
    // freadm, subtract '0' from everything using avx512, then whitespace is now negative/underflow so find the number
    // multithread this process so t

    // also multithread output too
    input_len = (int) fread(buffer, 1, buffer_len, stdin); // 20 is arbitrary number, assuming max 20 characters per number
    input_index = 0;
    // cast to intvec and subtract all by '0'

    int num_threads = omp_get_num_threads();
    int chunk_size = (input_len / CHAR_SIMD_STRIDE / num_threads) * CHAR_SIMD_STRIDE;
    // split string into num_threads chunks. if the boundary happens to land in the middle of an integer, move the start index to the left
    int chunk_indexes[num_threads+1];
    for (int i = 0; i < num_threads; i++) {
        chunk_indexes[i] = i*chunk_size;
        __builtin_prefetch(&buffer[chunk_indexes[i]-3]); // prefetch for next part
    }
    chunk_indexes[num_threads] = input_len;

    for (int i=0; i<num_threads; i++) {
        int start_index = chunk_indexes[i];
        while (buffer[start_index] != ' ' && buffer[start_index] != '\n') {
            start_index--;
        }
        chunk_indexes[i] = start_index+1;
    }

    int global_A_index=0;

    #pragma omp parallel for
    for (int i=0; i<num_threads; i++) {
        int start = chunk_indexes[i];
        int end = chunk_indexes[i+1];
        int end_simd_boundary = end - (end % CHAR_SIMD_STRIDE);

        int *tmp = new int[sizeof(int)*n*n];
        int tmp_index = 0;

        // start on current index, end on index+1
        for (int i=start; i<end_simd_boundary; i+=CHAR_SIMD_STRIDE) {
            charvec* tmp = reinterpret_cast<charvec*>(buffer + i);
            *tmp -= '0';
        }
        for (int i=end_simd_boundary; i<end; i++) {
            buffer[i] -= '0';
        }
        int current_index=0;
        int cur=0;
        auto readInt = [&]() -> bool {
            int cur=0;
            unsigned char c = buffer[input_index++];
            while(c == SPACE || c == NEWLINE && __builtin_expect(input_index<end, 0)) { // TODO: use builtin_expect
                c = buffer[input_index++];
            }
            if (__builtin_expect(input_index==end, 0)) {
                return false;
            }

            int is_negative = (c == NEGATIVE);
            c = buffer[input_index - 1 + is_negative];
            current_index += is_negative;

            int sign = 1 - (is_negative << 1);
            while (c <= 9) {
                cur = cur * 10 + (c - '0');
                c = buffer[input_index++];
            }
            cur *= sign;
            return true;
        };
        while (__builtin_expect(readInt(), 1)) {
            tmp[tmp_index++] = cur;
        }
        #pragma openmp barrier
        #pragma openmp ordered
        {

        };


        delete[] tmp;
    }



}

static const int divide_10_lookup[100] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9
};

static const int modulo_10_lookup[100] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9
};


static signed char *output_ptr = (signed char *) &buffer;
static inline __attribute__((always_inline)) void writeChar(signed char c) {
    *(output_ptr++) = c;
}

signed char SIGNED_NEGATIVE='-'-'0', SIGNED_SPACE=' '-'0', SIGNED_NEWLINE = '\n'-'0';

static __attribute__((always_inline)) inline void fastPrintMatrix(const intvec* _C, int n, int stride) {
    // use buffer for output
    const char beginning [] = "The resulting matrix C = A x B is:\n";
    fwrite(beginning, 1, strlen(beginning), stdout);

    const int* C = reinterpret_cast<const int*>(_C);

    for (int i=0; i<n; i++) {
        int baseind = i*stride;
        for (int j=0; j<n; j++) {
            int num = C[baseind + j];
            if (num < 0) {
                writeChar('-');
                num = -num;
            }

            char buf[12];
            int len = 0;

            if (__builtin_expect(num < 100, 1)) {
                if (num >= 10) {
                    buf[len++] = '0' + modulo_10_lookup[num];
                    num = divide_10_lookup[num];
                }
                buf[len++] = '0' + num;
            } else {
                do {
                    buf[len++] = '0' + num % 10;
                    num /= 10;
                } while (num);
            }
            for (int i=len-1; i>=0; --i) {
                writeChar(buf[i]);
            }

            writeChar(' ');
        }

        writeChar('\n');
    }

    fwrite(buffer, 1, (size_t) (output_ptr - (signed char*) buffer), stdout);
}
//
//void fastPrintMatrixParallel(const intvec* _C, int n, int stride)
//{
//    const char beginning[] = "The resulting matrix C = A x B is:\n";
//    fwrite(beginning, 1, strlen(beginning), stdout);
//
//    const int* C = reinterpret_cast<const int*>(_C);
//
//    // 1) Figure out how many threads we'll have
//    int nThreads = omp_get_max_threads();
//    {
//#pragma omp single
//        nThreads = omp_get_num_threads();
//    }
//
//    char**  threadBuffers  = new char*[nThreads];
//    size_t* threadBufSizes = new size_t[nThreads];
//
//#pragma omp parallel
//    {
//        // 3) Each thread calculates which rows it is responsible for
//        int tId = omp_get_thread_num();
//        int chunkSize = (n + nThreads - 1) / nThreads;  // "ceil" of n / nThreads
//        int startRow  = tId * chunkSize;
//        int endRow    = std::min(startRow + chunkSize, n);
//
//        // # of rows in this chunk
//        int rowCount  = endRow - startRow;
//        // # of total ints in this chunk = rowCount * n
//        // We'll allocate 20 bytes per integer "just to be safe" (sign + up to 10 digits + space, etc.)
//        size_t bufferSize = 20ULL * rowCount * n;
//        char* localBuffer = new char[bufferSize];
//        threadBuffers[tId] = localBuffer;
//
//        // 4) Convert the matrix rows into text, stored in localBuffer
//        char* ptr = localBuffer;
//        for (int i = startRow; i < endRow; i++) {
//            int baseInd = i * stride;
//            for (int j = 0; j < n; j++) {
//                int num = C[baseInd + j];
//
//                // Handle negative
//                if (num < 0) {
//                    *ptr++ = '-';
//                    num = -num;
//                }
//
//                // Convert integer -> decimal
//                char tmp[12];
//                int len = 0;
//
//                // quick path if < 100
//                if (__builtin_expect(num < 100, 1)) {
//                    if (num >= 10) {
//                        tmp[len++] = (char)('0' + modulo_10_lookup[num]);
//                        num = divide_10_lookup[num];
//                    }
//                    tmp[len++] = (char)('0' + num);
//                } else {
//                    while (num > 0) {
//                        tmp[len++] = (char)('0' + (num % 10));
//                        num /= 10;
//                    }
//                }
//                // Write out the digits in the correct order
//                while (len > 0) {
//                    *ptr++ = tmp[--len];
//                }
//                // Space after each number
//                *ptr++ = ' ';
//            }
//            // End of row: newline
//            *ptr++ = '\n';
//        }
//
//        // 5) Remember how many bytes this thread wrote
//        threadBufSizes[tId] = static_cast<size_t>(ptr - localBuffer);
//    } // end parallel region
//
//    // 6) Now print in the correct order: thread 0's chunk, then thread 1, ...
//    for (int t = 0; t < nThreads; ++t) {
//        fwrite(threadBuffers[t], 1, threadBufSizes[t], stdout);
//        delete[] threadBuffers[t]; // free each buffer after printing
//    }
//    delete[] threadBuffers;
//    delete[] threadBufSizes;
//}

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

    auto start = std::chrono::high_resolution_clock::now();


    // 1. Allocate memory
    START_TIMER // allocating memory and stuff
    buffer_len = (n*n*sizeof(int)*20 + CHAR_SIMD_STRIDE-1)/CHAR_SIMD_STRIDE * CHAR_SIMD_STRIDE;

    buffer = static_cast<unsigned char*>(std::aligned_alloc(alignment, buffer_len)); // 20 is arbitrary number, assuming max 20 characters per number
    if (buffer == nullptr) {
        throw std::bad_alloc();
    }
    output_ptr = reinterpret_cast<signed char*>(buffer);

    const int NUM_BLOCKS_COL = (n+kernel_size.second-1)/kernel_size.second * (kernel_size.second/BLOCK_SIZE); // pack BLOCK_SIZE ints per vector
    const int NUM_BLOCKS_ROW = (n+kernel_size.first-1)/kernel_size.first * kernel_size.first; // pack BLOCK_SIZE ints per vector

    // Create matrices A, B, and C (all n x n, blocked)
    intvec* A = intvec_alloc(NUM_BLOCKS_ROW*NUM_BLOCKS_COL); // [[NUM_BLOCKS_COL][NUM_BLOCKS_COL] ... ]
    intvec* B = intvec_alloc(NUM_BLOCKS_ROW*NUM_BLOCKS_COL);
    intvec* C = intvec_alloc(NUM_BLOCKS_ROW*NUM_BLOCKS_COL);

    END_TIMER("alloc");

    // 2. Optimized read input
//    START_TIMER // reading input
//
//    fastReadInput(A,B,n);
//
//    END_TIMER("read");


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

//    std::cerr << "N_KERNEL_COLS_PER_L3: " << N_KERNEL_COLS_PER_L3*kernel_size.second << "\n";
    const int N_KERNEL_ROWS_PER_L2 = (L2_CACHE_SIZE / (sizeof(int) * NUM_BLOCKS_COL * BLOCK_SIZE)) / kernel_size.first; assert(N_KERNEL_ROWS_PER_L2 > 0); // for 2048 matrix, block size=4, each row selection block takes 2048*4 = 8KB.
//    std::cerr << "N_KERNEL_ROWS_PER_L2: " << N_KERNEL_ROWS_PER_L2*kernel_size.first << "\n";
    const int N_KERNEL_ROWS_PER_L1 = L1_CACHE_SIZE / (N_KERNEL_COLS_PER_L3 * sizeof(int) * kernel_size.second) / kernel_size.first; assert(N_KERNEL_ROWS_PER_L1 > 0); // for 2048 matrix, block size=4, for a single column selection (block_size columns), we need block_size * block_size * 4 = 64B of data
//    std::cerr << "N_KERNEL_ROWS_PER_L1: " << N_KERNEL_ROWS_PER_L1*kernel_size.first << "\n";


    #pragma omp parallel for collapse(2)
    for (int i3=0; i3<NUM_BLOCKS_COL; i3+=N_KERNEL_COLS_PER_L3 * kernel_size.second / BLOCK_SIZE) { // select columns of B (of size 4*n) to go into the L3 cache
        for (int i2=0; i2<NUM_BLOCKS_ROW; i2+= N_KERNEL_ROWS_PER_L2 * kernel_size.first) { // select rows of A (of size 4*n) to go into the L2 cache
            for (int i1=0; i1<NUM_BLOCKS_ROW; i1+= N_KERNEL_ROWS_PER_L1 * kernel_size.first) { // select rows of B (of size 4*n) to go into the L1d cache

                // here we are only considering processing whole kernels, the dimensions should be multiples of kernel_size.
                // if the block is not complete, then the std::min will make sure we don't go out of bounds
                int i1_end = std::min(i1 + N_KERNEL_ROWS_PER_L1 * kernel_size.first, n);
                int i2_end = std::min(i2 + N_KERNEL_ROWS_PER_L2 * kernel_size.first, NUM_BLOCKS_ROW);
                int i3_end = std::min(i3 + N_KERNEL_COLS_PER_L3 * kernel_size.second / BLOCK_SIZE, NUM_BLOCKS_COL);

                for (int x=i2; x<i2_end; x+=kernel_size.first) { // select vertical kernel blocks of A
                    for (int y=i3; y<i3_end; y+= kernel_size.second / BLOCK_SIZE) { // select horizontal kernel blocks of B
                        kernel_v2(A, B, C, x, y, i1, i1_end, NUM_BLOCKS_COL);
                    }
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cerr << "Time in seconds: " << duration.count() / 1000000.0 << "\n";

//    std::cout << "The resulting matrix C = A x B is:\n";
//    for (int i = 0; i < n; ++i) {
//        for (int j = 0; j < n; ++j) {
//            std::cout << C[i*NUM_BLOCKS_COL+j/BLOCK_SIZE][j%BLOCK_SIZE] << " ";
//        }
//        std::cout << "\n";
//    }

    START_TIMER
    fastParallelPrintMatrix(C, n, NUM_BLOCKS_COL * BLOCK_SIZE);
    END_TIMER("print");
    return 0;
}
