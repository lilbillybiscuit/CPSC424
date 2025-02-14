/*******************************************************
* matrixmult.cpp
 *
 * Multiplies two square matrices of size n x n.
Time 1.66404s
 *******************************************************/

#include <iostream>
#include <vector>

#include <omp.h>

#define ALIGNMENT 256

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

int main() {
    int n;
    std::cin >> n;

    // Create matrices A, B, and C (all n x n)
    std::vector<std::vector<int>> A(n, std::vector<int>(n));
    std::vector<std::vector<int>> B(n, std::vector<int>(n));
    std::vector<std::vector<int>> C(n, std::vector<int>(n, 0));

    // Read matrix A
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cin >> A[i][j];
        }
    }

    // Read matrix B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cin >> B[i][j];
        }
    }

    // TODO: perform matrix multiplication A x B and write into C: C = A x B
    // YOUR CODE HERE
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            for (int k=0; k<n; k++) {
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }

    std::cout << "The resulting matrix C = A x B is:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << C[i][j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
