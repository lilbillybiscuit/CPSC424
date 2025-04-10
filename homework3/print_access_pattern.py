#!/usr/bin/env python3

def main():
    # Set N (you can modify this value)
    N = 10

    # Print the header with fields separated by tabs
    header = "i\tB[i]\tA[i]\tB[i+N]\tB[N-i]\tA[N-i]\tB[2*N-i]"
    print(header)

    # Loop for i in range(1, N) as in the C++ code
    for i in range(1, N):
        # For the first part, using 0-based indexing as shown in the example
        left_B = f"B[{i-1}]"
        left_A = f"A[{i-1}]"
        left_B_iN = f"B[{(i-1)+N}]"

        # For the second part, direct computation
        right_B = f"B[{N-i}]"
        right_A = f"A[{N-i}]"
        right_B_2Ni = f"B[{2*N-i}]"

        # Print all pieces separated by tabs
        print(f"{i}\t{left_B}\t{left_A}\t{left_B_iN}\t{right_B}\t{right_A}\t{right_B_2Ni}")

if __name__ == "__main__":
    main()
