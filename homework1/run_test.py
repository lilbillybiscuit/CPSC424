import numpy as np
import subprocess
import time

import sys
n=int(sys.argv[1]) if len(sys.argv) > 1 else 10
shouldprint = bool(sys.argv[2]) if len(sys.argv) > 2 else False

A = np.random.randint(0, 10, size=(n,n))
B = np.random.randint(0, 10, size=(n,n))

C = np.dot(A,B)

s = str(n) + "\n"
s += "\n".join([" ".join([str(x) for x in row]) for row in A]) + "\n"
s += "\n".join([" ".join([str(x) for x in row]) for row in B]) + "\n"

compile_cmd = ["g++", "-std=c++17", "-O3", "-fopenmp", "-mavx2", "-mavx512f",   "-o", "matrixmult", "matrixmult.cpp"]
res = subprocess.run(compile_cmd)
if res.returncode != 0:
    print("Compilation failed")
    exit()
total_time = 0.0
num_runs = 5
for i in range(num_runs):
    start_time = time.time()
    output = subprocess.run(["./matrixmult"], input=s.encode(), capture_output=True)
    end_time = time.time()
    print("Time taken: ", end_time - start_time)
    total_time += end_time - start_time

    temp_output = output.stdout.decode().split("\n")
    print(output.stderr.decode().split("\n")[0])
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
        exit(0)



print("Passed")
print("Average time: ", total_time/num_runs)




