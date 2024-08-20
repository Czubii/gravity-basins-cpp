#include <iostream>
#include <cuda_runtime.h>

using namespace std;

namespace cr{


const int ARRAY_SIZE = 10;
const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

// Kernel function to add elements of two arrays
__global__ void addArrays(float *a, float *b, float *c) {
    int index = threadIdx.x;
    if (index < ARRAY_SIZE) {
        c[index] = a[index] + b[index];
    }
}

void test() {
    // Host arrays
    float h_a[ARRAY_SIZE], h_b[ARRAY_SIZE], h_c[ARRAY_SIZE];
    
    // Initialize host arrays
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    // Device arrays
    float *d_a, *d_b, *d_c;
    
    // Allocate memory on the device
    cudaMalloc((void **)&d_a, ARRAY_BYTES);
    cudaMalloc((void **)&d_b, ARRAY_BYTES);
    cudaMalloc((void **)&d_c, ARRAY_BYTES);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // Launch kernel with 1 block and ARRAY_SIZE threads
    addArrays<<<1, ARRAY_SIZE>>>(d_a, d_b, d_c);

    // Copy result from device to host
    cudaMemcpy(h_c, d_c, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Results:" << std::endl;
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}

}