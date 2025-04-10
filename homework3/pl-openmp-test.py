#!/usr/bin/env python3
import subprocess
import struct
import sys
import numpy as np
import os

def compile_cpp():
    """
    Compiles the C++ source file (pl-openmp.cpp) to produce the executable 'openmp_test'.
    Adjust the compile command as needed for your platform.
    """
    # Example for Linux:

    # search for file in current directory starting with prefix sys.argv[1] and ending with .cpp
    # if no file is found, use pl-openmp.cpp
    # if there are multiple, print out a red error message
    filename = "pl-openmp.cpp"
    print(sys.argv)
    if len(sys.argv) == 2:
        files = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith('.cpp') and f.startswith(sys.argv[1])]
        if len(files) == 0:
            print("\033[91mError: No file found starting with prefix", sys.argv[1], "and ending with .cpp\033[0m")
            sys.exit(1)
        elif len(files) > 1:
            print("\033[91mError: Multiple files found starting with prefix", sys.argv[1], "and ending with .cpp\033[0m")
            sys.exit(1)

        filename = files[0]
    elif len(sys.argv) > 2:
        # print error
        print("\033[91mError: Incorrect number of arguments\033[0m")

    compile_command = ["g++-12", "-O3", "-fopenmp", "-o", "openmp_test", filename]
    # If you are on macOS and have OpenMP installed via Homebrew, you might need:
    # compile_command = ["g++", "-O3", "-Xpreprocessor", "-fopenmp", "-lomp", "-o", "openmp_test", "pl-openmp.cpp"]

    try:
        print(f"Compiling C++ code ({filename})...")
        subprocess.run(compile_command, check=True)
        print("Compilation successful.\n")
    except subprocess.CalledProcessError as e:
        print("Compilation failed!")
        sys.exit(1)

def parse_cpp_output(output):
    """
    Extract the final sum and the time taken from the C++ output.
    Expected output is two lines, for example:
        "Final sum: 1613\nTime taken: 0.001234 seconds\n"
    Returns a tuple: (final_sum, time_taken)
    """
    if isinstance(output, bytes):
        output = output.decode("utf-8")
    final_sum = None
    time_taken = None
    for line in output.splitlines():
        if line.startswith("Final sum:"):
            try:
                # Convert via float then int to handle scientific notation.
                final_sum = int(float(line[len("Final sum:"):].strip()))
            except Exception as e:
                raise ValueError(f"Error converting final sum to int: {e}")
        elif line.startswith("Time taken:"):
            try:
                # Remove the prefix and "seconds" suffix if present.
                s = line[len("Time taken:"):].strip()
                if s.endswith("seconds"):
                    s = s[:-len("seconds")].strip()
                time_taken = float(s)
            except Exception as e:
                raise ValueError(f"Error converting time taken to float: {e}")
    if final_sum is None:
        raise ValueError("Final sum not found in output")
    if time_taken is None:
        raise ValueError("Time taken not found in output")

    return final_sum, time_taken

def run_cpp_executable(N, A, B, executable="./openmp_test"):
    """
    Pack the data in binary (N as int64, then A and B as int64 arrays) and run the C++ executable.
    Returns the final sum (an integer) parsed from the executable's output.
    """
    # Pack N as a 64-bit signed integer.
    data = struct.pack("q", N)
    # Pack A and B as int64 arrays.
    data += A.astype(np.int64).tobytes()
    data += B.astype(np.int64).tobytes()

    proc = subprocess.run([executable],
                          input=data,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print("Error running executable:")
        print(proc.stderr.decode("utf-8"))
        sys.exit(1)
    print(proc.stderr.decode("utf-8"))
    try:
        result = parse_cpp_output(proc.stdout)
    except Exception as e:
        print("Error parsing result:", e)
        sys.exit(1)
    return result

def simulate_algorithm_py(A, B):
    """
    Pure Python simulation of the algorithm using integer arithmetic.
    A and B are assumed to be numpy arrays of int64.
    """
    N = len(A)
    A_sim = A.copy()
    B_sim = B.copy()

    # Step 1: A[i] += i
    A_sim += np.arange(N, dtype=np.int64)

    # Step 2: B[i] += i%2 for first N elements
    B_sim[:N] += np.arange(N, dtype=np.int64) % 2

    # Step 3a: B[i] += A[i] + B[i+N] for first N elements
    B_sim[:N] += A_sim + B_sim[N:]

    # Step 3b: B[i] += 2*A[i-N] + B[i-N] for last N elements
    B_sim[N:] += 2 * A_sim + B_sim[:N]

    # Step 4: A[i] = (B[i] + B[N-i])//2 (using integer division)
    # Create index array for B[N-i] where i ranges 0..N-1
    reverse_indices = N - np.arange(N, dtype=np.int64)
    A_sim[:] = (B_sim[:N] + B_sim[reverse_indices]) // 2

    return np.sum(A_sim)


def main():
    # First, compile the C++ code.
    compile_cpp()

    # Optionally, set a random seed for reproducibility.
    # np.random.seed(42)

    test_sizes = [10, 100, 1000, 10000, 1000000, 10000000, 100000000]
    # With integer arithmetic we expect an exact match.
    all_passed = True
    executable = "./openmp_test"  # Adjust the path if needed.


    for N in test_sizes:
        # Generate random integers in the range [1, 100].
        A = np.random.randint(1, 101, size=(N,)).astype(np.int64)
        B = np.random.randint(1, 101, size=(2 * N,)).astype(np.int64)

        expected = simulate_algorithm_py(A, B)
        # print(f"Starting test {N} with expected sum {expected}...")
        result = run_cpp_executable(N, A, B, executable=executable)
        answer = result[0]
        seconds = result[1]

        print(f"Test with N = {N}: expected {expected}, got {answer}; seconds: {seconds}")
        if answer == expected:
            print("  PASS")
        else:
            print(f"  FAIL (difference {expected - answer}, {100*abs(expected - answer)/N:.2f}% of N)")
            all_passed = False

    if not all_passed:
        sys.exit(1)

if __name__ == '__main__':
    main()
