// openmp_test.cpp
#include <iostream>
#include <cstdint>
#include <vector>
#include <omp.h>
#include <iomanip>

const size_t ALIGNMENT = 64;


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


typedef std::vector<int64_t, AlignedVectorAllocator<int64_t, ALIGNMENT>> alignedvec64;
int64_t solve(int64_t N, alignedvec64& A, alignedvec64& B) {
    int64_t sum = B[N]+A[0]+B[0];
    sum+=B[0] + 3*A[0] + 2*B[N];
    sum/=2;

    #pragma omp parallel for schedule(static) reduction(+:sum)
    for (int64_t i = 1; i < (N+1)/2; i++) {
        sum += ((B[i] + A[i] + B[i+N] ) + (B[N-i] + A[N-i] + B[2*N-i]))&~1;;
    }

    if (!(N&1u)) {
        const int tmp = N/2;
        sum += ((B[tmp] + A[tmp] + B[tmp+N] ) + (B[tmp] + A[tmp] + B[2*N-tmp]))/2;;
    }

    sum += (N*(N-1) + N)/2;
    return sum;
}

// typedef std::vector<int64_t, AlignedVectorAllocator<int64_t, ALIGNMENT>> alignedvec64;
// int64_t solve(int64_t N, alignedvec64& A, alignedvec64& B) {
//     // Compute special term for i = 0, with its per-iteration division.
//     int64_t special = ((B[N] + A[0] + B[0]) + (B[0] + 3*A[0] + 2*B[N])) / 2;
//
//     int64_t sum_forward = 0;
//     // Loop 1: Process forward contributions for i = 1 to N-1.
//     for (int64_t i = 1; i < N; i++) {
//         // Compute the forward half term and divide by 2.
//         int64_t forward_term = (B[i] + A[i] + B[i+N] + i + (i & 1));
//         sum_forward += forward_term / 2;
//     }
//
//     int64_t sum_reverse = 0;
//     // Loop 2: Process re-indexed reverse contributions.
//     // Re-index: Let j = N - i, so i runs from 1 to N-1 in forward order.
//     for (int64_t j = 1; j < N; j++) {
//         int64_t reverse_term = (B[j] + A[j] + B[N+j] + (N - j) + ((N - j) & 1));
//         sum_reverse += reverse_term / 2;
//     }
//
//     // Combine all contributions.
//     return special + sum_forward + sum_reverse;
// }

int main() {
    // Read the integer N (as a 64-bit signed integer) from standard input in binary.
    int64_t N;
    std::cin.read(reinterpret_cast<char*>(&N), sizeof(N));
    if (!std::cin) {
        std::cerr << "Error reading N from input." << std::endl;
        return 1;
    }
    int N_aligned = ((N-1)/ALIGNMENT+1) * ALIGNMENT;

    // Allocate arrays:
    // A is a int64 array of length N.
    // B is a int64 array of length 2*N.
    alignedvec64 A(N_aligned); // align on 64-byte boundary, size is multiple of 64
    alignedvec64 B(2 * N_aligned);

    // Read array A from binary input.
    std::cin.read(reinterpret_cast<char*>(A.data()), N * sizeof(int64_t));
    if (!std::cin) {
        std::cerr << "Error reading A from input." << std::endl;
        return 1;
    }

    // Read array B from binary input.
    std::cin.read(reinterpret_cast<char*>(B.data()), 2 * N * sizeof(int64_t));
    if (!std::cin) {
        std::cerr << "Error reading B from input." << std::endl;
        return 1;
    }

    // Get the start time.
    double start_time = omp_get_wtime();

    // TODO: your code here. Return the sum in the sum double.
    int64_t sum = solve(N, A, B);

    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;

    // Output the results in the expected format.
    std::cout << std::fixed << std::setprecision(10) <<
        "Final sum: " << sum << std::endl;
    std::cout << "Time taken: " << elapsed << " seconds" << std::endl;

    return 0;
}
