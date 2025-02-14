/*******************************************************
* matrixmult.cpp
 *
 * Multiplies two square matrices of size n x n.
 * Author: Bill Qian, with credit going to Andrew Tran for his help and ideas
Read time: 0.17s
Matrix Mult time: 0.38s
Write time: 0.11s
 *******************************************************/

#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>

#include <omp.h>
#include <cstring>

// --- Start Timer Code
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
// --- End Timer Code

const std::pair<int,int> kernel_size = {6, 32}; // second number should be a multiple of BLOCK_SIZE
const   int BLOCK_SIZE = 16;
const   int alignment = 64;
const   int VECTOR_SIZE = sizeof(int)*BLOCK_SIZE;

const   int CHAR_SIMD_STRIDE = 64; // on 512-bit registers, 512 bits = 64 bytes = 64 chars

//const   int L3_CACHE_SIZE = 1024*1024*2; // 32 MB (unused due to manual definition to try to reduce chances of irregular kernel/block sizing)
const   int L2_CACHE_SIZE = 1024*1024; // 1024 KB
const   int L1_CACHE_SIZE = 128*1024; // 128 KB (incorrect, but did not catch this earlier whoopsies)

const int N_KERNEL_COLS_PER_L3 = 48;

typedef int intvec __attribute__ ((vector_size(VECTOR_SIZE))); // 32 bytes = 8 integers (8 * 4 bytes/int)
typedef char charvec __attribute__ ((vector_size(CHAR_SIMD_STRIDE))); // 64 bytes = 64 chars (64 * 1 byte/char)
typedef signed char charvecmini __attribute__ ((vector_size(16)));

static __attribute__((always_inline)) inline intvec *intvec_alloc(std::size_t n_bytes) {
    intvec* res =  static_cast<intvec*>(std::aligned_alloc(alignment, n_bytes*sizeof(intvec)));
    if (res == nullptr) {
        throw std::bad_alloc();
    }
    return res;
}

/*
 * Kernel function
 * updates C[x:x+kernel_size.first][y:y+kernel_size.second/BLOCK_SIZE]
 * multiply A[x:x+kernel_size.first][l:r] with B[l:r][y:y+kernel_size.second/BLOCK_SIZE]
 * x and y determine the continuous kernel_size block that is modified in C. They do not have to be
 */
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

    // jank on jank on jank sequence of macros
    // the original code (eg. when expanded) is located here: https://github.com/lilbillybiscuit/random-code/blob/main/parallel_sample/pset1_kernel_v2.1.cpp
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


static __attribute__((always_inline)) inline int fastReadIntDirect() {
    // only useful for reading the first integer n before sending it with the fread(1GB buffer, stdin)
    int result = 0;
    int c = fgetc(stdin);

    while (c == ' ' || c == '\n') {
        c = fgetc(stdin);
    }

    bool negative = (c == '-');
    if (negative) {
        c = fgetc(stdin);
    }

    while (c >= '0' && c <= '9') {
        result = result * 10 + (c - '0');
        c = fgetc(stdin);
    }

    return negative ? -result : result;
}


inline __attribute__((always_inline))
void fastReadInt(int &number, unsigned char* buffer, int &pos, const int end)
{
    // standard integer reading is too slow so we're going to it manually.
    // this is purely hyper-hyperoptimizing
    // credit goes to Andrew Tran for the idea and implementation details
    // here goes...
    int c = buffer[pos++];
    while (__builtin_expect((c == ' ' || c == '\n'), 1)) {
        if (__builtin_expect(pos >= end, 0)) {
            // we're at chunk's end or beyond
            number = 0;
            return;
        }
        c = buffer[pos++];
    }
    // check sign
    int neg = (c == '-');
    if (neg) {
        if (__builtin_expect(pos >= end, 0)) { number = 0; return; }
        c = buffer[pos++];
    }
    int sign = 1 - (neg << 1); // +1 or -1

    number = 0;

    if ((unsigned)(c - '0') <= 9) {
        number = number * 10 + (c - '0');
        if (__builtin_expect(pos >= end, 0)) { number *= sign; return; }
        c = buffer[pos++];
        if ((unsigned)(c - '0') <= 9) {
            number = number * 10 + (c - '0');
            if (__builtin_expect(pos >= end, 0)) { number *= sign; return; }
            c = buffer[pos++];
            if ((unsigned)(c - '0') <= 9) {
                number = number * 10 + (c - '0');
                if (__builtin_expect(pos >= end, 0)) { number *= sign; return; }
                c = buffer[pos++];
                if ((unsigned)(c - '0') <= 9) {
                    number = number * 10 + (c - '0');
                    if (__builtin_expect(pos >= end, 0)) { number *= sign; return; }
                    c = buffer[pos++];
                }
            }
        }
    }

    while (__builtin_expect((unsigned)(c - '0') <= 9, 0)) {
        number = number * 10 + (c - '0');
        if (__builtin_expect(pos >= end, 0)) { break; }
        c = buffer[pos++];
    }

    number *= sign;
}


static unsigned char* buffer = nullptr;
static int input_index = 0, input_len = 0, buffer_len = 0;
const unsigned char SPACE = ' '-'0', NEWLINE = '\n' - '0', NEGATIVE = '-' - '0';
static void fastParallelReadInput(intvec *__restrict__ A,
                                  intvec *__restrict__ B,
                                  int n,
                                  int stride)
{
    // "what am i doing with my life" counter: 9
    // read the input via fread, the parse it in parallel
    // utilizes alien math, too much trial and error, and praying to the rat gods that things don't break
    // 1. read all data
    input_len = std::fread(buffer, 1, buffer_len, stdin);
    if (input_len <= 0) {
        std::cerr << "[fastParallelReadInput] no input data.\n";
        return;
    }

    const int total_ints = 2 * n * n;

    std::vector<int> temp(total_ints);

    // 2. decide #threads:
    int max_threads = omp_get_max_threads();
    int num_threads = std::max(1, std::min(max_threads, input_len / 2000));

    // 3. build chunk offsets
    std::vector<int> chunkStart(num_threads+1, 0);
    int chunk_size = (input_len + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; t++) {
        chunkStart[t] = t * chunk_size;
    }
    chunkStart[num_threads] = input_len;

    for (int t = 1; t < num_threads; t++) {
        int idx = chunkStart[t];
        while (idx < input_len && ((buffer[idx] >= '0' && buffer[idx] <= '9') || (buffer[idx] == '-'))) {
            idx++;
        }
        chunkStart[t] = std::min(idx, input_len);
    }

    // partialCounts[t] = how many ints thread t parsed. will be combined together in the end on O(thread) time (16 threads -> log(therad)) time not worth it
    std::vector<int> partialCounts(num_threads, 0);

    // 4. each thread parses its own chunk
#pragma omp parallel num_threads(num_threads)
    {
        int t = omp_get_thread_num();
        int start = chunkStart[t];
        int end   = chunkStart[t+1];

        int localCapacity = (end - start)/2 + 10;
        int* localArray = new int[localCapacity];

        int localCount = 0;
        int pos = start;
        while (pos < end) {
            // parse one int
            int number;
            fastReadInt(number, buffer, pos, end);
            if (pos > end + 2) {
                break;
            }
            // store
            localArray[localCount++] = number;
            if (__builtin_expect(localCount >= localCapacity, 0)) {
                break;
            }
        }

        partialCounts[t] = localCount;

        // barrier so partialCounts is ready
#pragma omp barrier
        // single thread does prefix sum - ~10-32 operations depending on system
        // calculates the position of where each thread's data will be written
#pragma omp single
        {
            for (int i = 1; i < num_threads; i++) {
                partialCounts[i] += partialCounts[i-1];
            }
        }

        // figure out where to write in "temp"
        int offset = (t == 0) ? 0 : partialCounts[t-1];
        int limit  = std::min(offset + localCount, total_ints);
        for (int i = offset; i < limit; i++) {
            temp[i] = localArray[i - offset];
        }

        delete[] localArray;
    }

    // how many we parsed in total
    int parsedTotal = partialCounts.empty() ? 0 : partialCounts[num_threads-1];
    if (parsedTotal < total_ints) {
        std::cerr << "[Warn] parsed " << parsedTotal << " < needed " << total_ints << "\n";
        // we can fill the remainder with zero, or do nothing
        for (int i = parsedTotal; i < total_ints; i++) {
            temp[i] = 0;
        }
    }

    // 4. the first n*n go to A, next n*n to B
    int haveA = std::min(parsedTotal, n*n);
    int haveB = std::max(0, std::min(parsedTotal - haveA, n*n));

    // fill A in parallel
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int idxA = i*n + j;
            if (idxA < haveA) {
                A[ i*stride + (j/BLOCK_SIZE) ][ j % BLOCK_SIZE ] = temp[idxA];
            } else {
                A[ i*stride + (j/BLOCK_SIZE) ][ j % BLOCK_SIZE ] = 0;
            }
        }
    }

    // bill B in parallel
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int idxB = n*n + i*n + j;
            if (idxB < n*n + haveB) {
                B[ i*stride + (j/BLOCK_SIZE) ][ j % BLOCK_SIZE ] = temp[idxB];
            } else {
                B[ i*stride + (j/BLOCK_SIZE) ][ j % BLOCK_SIZE ] = 0;
            }
        }
    }
}

// credit to Andrew Tran for this invention
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


signed char SIGNED_NEGATIVE='-'-'0', SIGNED_SPACE=' '-'0', SIGNED_NEWLINE = '\n'-'0';

static void fastWriteInt(signed char * &output_buffer, int &num) {
    // standard write is slow due to divides, modulos, and other expensive branch operations.
    // credit to Andrew Tran for the idea of modulo/divide lookup tables, and some further optimizations to this function
    // here goes...
    charvecmini buf = charvecmini{};
    int len = 0;
    bool negative = false;
    if (num < 0) {
        negative = true;
        num = -num;
    }
    if (__builtin_expect(num < 100, 1)) {
        if (num >= 10) {
            buf[len++] =  modulo_10_lookup[num];
            num = divide_10_lookup[num];
        }
        buf[len++] = num;
    } else {
        do {
            buf[len++] = num % 10;
            num /= 10;
        } while (num);
    }

    if (negative) {
        buf[len++] = SIGNED_NEGATIVE;
    }

    buf += '0';
    buf[len] = '\0';

    for (int i=len-1; i>=0; i--) {
        *output_buffer++ = buf[i];
    }
}

static __attribute__((always_inline)) inline void fastParallelPrintMatrix(const intvec* _C, int n, int stride) {
    // use buffer for output
    const char beginning [] = "The resulting matrix C = A x B is:\n";
    fwrite(beginning, 1, strlen(beginning), stdout);

    const int* C = reinterpret_cast<const int*>(_C);


    int num_threads = std::min(omp_get_max_threads(), n);
    int *chunk_barriers = new int[num_threads+1];
    chunk_barriers[0]=0;
    for (int i=1; i<num_threads; i++) {
        chunk_barriers[i] = i * (n/num_threads);
    }
    chunk_barriers[num_threads] = n;

    #pragma omp parallel for ordered
    for (int thread_id=0; thread_id < num_threads; thread_id++) {
        int start = chunk_barriers[thread_id];
        int end = chunk_barriers[thread_id+1];

        int thread_buffer_size = (end-start)*20*n;
        thread_buffer_size = (thread_buffer_size + alignment-1)/alignment*alignment;
        auto * thread_buffer = static_cast<signed char*>(std::aligned_alloc(alignment, thread_buffer_size));
        auto * thread_pointer = thread_buffer;

        for (int i=start; i<end; i++) {
            int baseind = i*stride;
            for (int j=0; j<n; j++) {
                int num = C[baseind + j];
                fastWriteInt(thread_pointer, num);
                // add space to thread buffer
                *(thread_pointer++)= ' ';
            }
            *(thread_pointer++) = '\n';
        }
        *thread_pointer='\0';

        #pragma omp ordered
        {
            fwrite(thread_buffer, 1, (size_t) (thread_pointer - (signed char*) thread_buffer), stdout);
        };
        std::free(thread_buffer);
    };

    delete[] chunk_barriers;
}



#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx512f")))
#else
#endif
int main() {
    assert(kernel_size.second % BLOCK_SIZE==0);
    assert(N_KERNEL_COLS_PER_L3 > 0);
    int n = fastReadIntDirect();
    auto start = std::chrono::high_resolution_clock::now();

    // 1. allocate memory
    START_TIMER // allocating memory and stuff
    buffer_len = (n*n*sizeof(int)*20 + CHAR_SIMD_STRIDE-1)/CHAR_SIMD_STRIDE * CHAR_SIMD_STRIDE;

    buffer = static_cast<unsigned char*>(std::aligned_alloc(alignment, buffer_len)); // 20 is arbitrary number, assuming max 20 characters per number
    if (buffer == nullptr) {
        throw std::bad_alloc();
    }

    // 1a. calculate dimensions
    const int NUM_BLOCKS_COL = (n+kernel_size.second-1)/kernel_size.second * (kernel_size.second/BLOCK_SIZE); // pack BLOCK_SIZE ints per vector
    const int NUM_BLOCKS_ROW = (n+kernel_size.first-1)/kernel_size.first * kernel_size.first; // pack BLOCK_SIZE ints per vector

    // create matrices A, B, and C (all n x n, blocked)
    intvec* A = intvec_alloc(NUM_BLOCKS_ROW*NUM_BLOCKS_COL); // [[NUM_BLOCKS_COL][NUM_BLOCKS_COL] ... ]
    intvec* B = intvec_alloc(NUM_BLOCKS_ROW*NUM_BLOCKS_COL);
    intvec* C = intvec_alloc(NUM_BLOCKS_ROW*NUM_BLOCKS_COL);
    memset(C, 0, NUM_BLOCKS_ROW*NUM_BLOCKS_COL*sizeof(intvec));

    END_TIMER("alloc");

    // 2. optimized read input
    START_TIMER // reading input
    fastParallelReadInput(A, B, n, NUM_BLOCKS_COL);
    END_TIMER("read");

    // 2a. calculate block constants
    const int N_KERNEL_ROWS_PER_L2 = (L2_CACHE_SIZE / (sizeof(int) * NUM_BLOCKS_COL * BLOCK_SIZE)) / kernel_size.first; assert(N_KERNEL_ROWS_PER_L2 > 0); // for 2048 matrix, block size=4, each row selection block takes 2048*4 = 8KB.
    const int N_KERNEL_ROWS_PER_L1 = L1_CACHE_SIZE / (N_KERNEL_COLS_PER_L3 * sizeof(int) * kernel_size.second) / kernel_size.first; assert(N_KERNEL_ROWS_PER_L1 > 0); // for 2048 matrix, block size=4, for a single column selection (block_size columns), we need block_size * block_size * 4 = 64B of data


    //    std::cerr << "N_KERNEL_COLS_PER_L3: " << N_KERNEL_COLS_PER_L3*kernel_size.second << "\n";
    //    std::cerr << "N_KERNEL_ROWS_PER_L2: " << N_KERNEL_ROWS_PER_L2*kernel_size.first << "\n";
    //    std::cerr << "N_KERNEL_ROWS_PER_L1: " << N_KERNEL_ROWS_PER_L1*kernel_size.first << "\n";


    // 3. multiply
    START_TIMER
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

    END_TIMER("multiply");

    // 4. print
    START_TIMER
    fastParallelPrintMatrix(C, n, NUM_BLOCKS_COL * BLOCK_SIZE);
    END_TIMER("print");
    return 0;
}
