import numpy as np
import subprocess
import time
from scipy.linalg.blas import sgemm

import argparse

parser = argparse.ArgumentParser(description='Process some arguments')
parser.add_argument('-n', '--number', type=int, default=10,
                    help='An integer value (default: 10)')
parser.add_argument('-p', '--print', action='store_true',
                    help='Whether to print or not')
parser.add_argument('-N', type=str, default="matrixmult.cpp",
                    help='Name of the executable file (default: matrixmult)')

args = parser.parse_args()

n = args.number
shouldprint = args.print
program_name = args.N
print("Initializing")
A = np.random.randint(-20, 20, size=(n,n))
B = np.random.randint(-20, 20, size=(n,n))
print("Multipliying Matrix")
# C = np.dot(A,B)
t = time.time()
C = sgemm(alpha=1.0, a=A, b=B)
end = time.time()
print("Matrix Multiplied in", end - t)



s = str(n) + "\n"
s += "\n".join([" ".join([str(x) for x in row]) for row in A]) + "\n"
s += "\n".join([" ".join([str(x) for x in row]) for row in B]) + "\n"

with open("tempfilecontent_100neg.txt", "w") as f:
    f.write(s)


compile_cmd = ["g++-12", "-std=c++17", "-O3", "-fopenmp", "-o", "matrixmult", f"{program_name}"]
res = subprocess.run(compile_cmd)
if res.returncode != 0:
    print("Compilation failed")
    exit()
total_time = 0.0
num_runs = 5

output = subprocess.run(["./matrixmult"], input=s.encode(), capture_output=True)
for i in range(num_runs):
    start_time = time.time()
    output = subprocess.run(["./matrixmult"], input=s.encode(), capture_output=True)
    end_time = time.time()
    print("Time taken: ", end_time - start_time)
    total_time += end_time - start_time

    temp_output = output.stdout.decode().split("\n")
    temp_stderr = output.stderr.decode()
    if len(temp_stderr) > 0:
        print(output.stderr.decode(), end="" if temp_stderr[-1] == "\n" else "\n")

    output = temp_output[1:]
    output = np.array([list(map(int, row.split())) for row in output if row])
    if not np.allclose(C, output):
        print("Failed")
        if shouldprint:
            print("Input A:")
            print(A)
            print("Input B:")
            print(B)
            print("Expected output:")
            print(C)
            print("Your output:")
            print(output)

            # print the first 10 differences when scanning left to right, top to bottom
            for i in range(min(10, len(output))):
                for j in range(min(10, len(output[0]))):
                    if C[i][j] != output[i][j]:
                        print(f"Difference at ({i}, {j}): expected {C[i][j]}, got {output[i][j]}")
                        break
        exit(0)


total_computations = 2 * n**3
print("Passed")
print("Average time         =", total_time/num_runs)
print("Total computations   =", total_computations)
print("GOPS/sec             ≈", total_computations / (total_time/num_runs) / 10**9)

# theoretical arithmetic ops = 8 (simd) * cycles_per_sec * threads
# import os
# # if macos, use something else
# # if linux, use cat /proc/cpuinfo
# if os.name == 'posix':
#     cpu_cycles_per_sec = int(os.popen("sysctl -n hw.cpufrequency").read().strip())
#     num_threads = int(os.popen("sysctl -n hw.ncpu").read().strip())
# else:
#     # linux
#     cpu_cycles_per_sec = int(os.popen("cat /proc/cpuinfo | grep 'cpu MHz' | awk '{print $4}'").read().strip()) * 10**6
#     num_threads = int(os.popen("cat /proc/cpuinfo | grep 'core id' | wc -l").read().strip())
# theoretical_arithmetic_ops = 8 * cpu_cycles_per_sec * num_threads
# print("Theoretical arithmetic ops: ", theoretical_arithmetic_ops)




