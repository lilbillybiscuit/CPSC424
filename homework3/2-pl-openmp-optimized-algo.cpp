// openmp_test.cpp
#include <iostream>
#include <cstdint>
#include <vector>
#include <omp.h>
#include <iomanip>

const size_t ALIGNMENT = 64;

int64_t solve(int64_t N, std::vector<int64_t>& A, std::vector<int64_t>& B) {
    int64_t sum = B[N]+A[0]+B[0];
    sum+=B[0] + 3*A[0] + 2*B[N];
    sum/=2;

#pragma omp parallel for schedule(static) reduction(+:sum)
#pragma openmp simd reduction(+:sum)
    for (int64_t i = 1; i < N; i++) {
        sum += ((B[i] + A[i] + B[i+N] + i + i%2) + (B[N-i] + A[N-i] + B[2*N-i] + i+ i%2))/2;
    }
    return sum;
}
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
    std::vector<int64_t> A(N_aligned);
    std::vector<int64_t> B(2 * N_aligned);

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
