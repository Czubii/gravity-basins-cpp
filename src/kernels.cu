

#include <cuda_runtime.h>
#include <iostream>
#include <iterator>
#include <queue>

#include "kernels.cuh"
#include "parameters.h"

using namespace std;

namespace kernel {

#define BLOCK_SIZE 2
#define BLOCK_SIZE_Z 8

// Fast inverse square root function
__device__ float fastInvSqrt(float &number) {
  if (number <= 0)
    return 0;
  float x = number;
  float x_half = 0.5f * x;
  int32_t i = *reinterpret_cast<int32_t *>(&x);
  i = 0x5f3759df - (i >> 1);
  x = *reinterpret_cast<float *>(&i);
  x = x * (1.5f - (x_half * x * x));
  return x;
}

// Fast square root function
__device__ float fastSqrt(float &number) { return 1.0f / fastInvSqrt(number); }

__device__ float getDistanceSquared(float &x1, float &y1, float &x2,
                                    float &y2) {
  return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

__device__ kernel::vector2f calculateGravityForce(float &x, float &y,
                                                  kernel::StaticBody &body) {
  float distanceSquared = kernel::getDistanceSquared(x, y, body.x, body.y);
  float distance = fastSqrt(distanceSquared);
  float dirX = (body.x - x) / distance;
  float dirY = (body.y - y) / distance;

  float magnitude =
      GRAVITY_CONSTANT * body.mass * PARTICLE_MASS / distanceSquared;

  return {dirX * magnitude, dirY * magnitude};
}

__device__ bool collidesWithBody(kernel::StaticBody &body,
                                 kernel::vector2f &pos) {
  float distanceSquared = getDistanceSquared(body.x, body.y, pos.x, pos.y);
  if (distanceSquared <= body.radius * body.radius)
    return true;

  return false;
}

__global__ void renderBasins(unsigned char *outputPixelArray,
                             size_t outputWidth, size_t outputHeight,
                             kernel::StaticBody *staticBodies,
                             unsigned char numStaticBodies, float renderScale,
                             int maxSteps = 10000) {
  extern __shared__ kernel::particleData sharedParticles[];

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int index2D = ((y * outputWidth + x));
  int insideBlockIndex2D = (threadIdx.y * blockDim.y) + threadIdx.x;

  if (x >= outputWidth || y >= outputHeight || z >= numStaticBodies) {
    return;
  }
  kernel::particleData &particle = sharedParticles[insideBlockIndex2D];
  if (z == 0) {
    particle.pos = {x / renderScale, y / renderScale};
    particle.vel = {0, 0};
    particle.terminated = false;
  }

  __syncthreads();

  kernel::StaticBody body = staticBodies[z];
  kernel::vector2f force = {0, 0};

  for (int step = 0; step < maxSteps; step++) {

    if (collidesWithBody(body, particle.pos)) {
      atomicExch(&particle.terminated, 1);
      outputPixelArray[index2D * 4] = body.color.r;
      outputPixelArray[index2D * 4 + 1] = body.color.g;
      outputPixelArray[index2D * 4 + 2] = body.color.b;
      outputPixelArray[index2D * 4 + 3] = body.color.a;
    }

    __syncthreads();
    if (particle.terminated != 0)
      return;

    force = calculateGravityForce(particle.pos.x, particle.pos.y, body);
    force.x = force.x / PARTICLE_MASS;
    force.y = force.y / PARTICLE_MASS;

    atomicAdd(&particle.vel.x, force.x);
    atomicAdd(&particle.vel.y, force.y);

    __syncthreads();

    if (z == 0) {
      particle.pos.x += particle.vel.x;
      particle.pos.y += particle.vel.y;
    }

    __syncthreads();
  }
}

unsigned char *testRender(int outputWidth, int outputHeight,
                          kernel::StaticBody *h_staticBodies,
                          unsigned char numStaticBodies, float renderScale) {
  if (numStaticBodies > BLOCK_SIZE_Z) {
    std::cout
        << "TO MANY STATIC BODIES. LIMIT IS" << BLOCK_SIZE_Z
        << " FOR CUDA RENDERING. CHANGE IT BY MODIFING BLOCK_SIZE_Z MACRO";
    return 0;
  }
  // ------ Alocate memory ------

  // output pixel array

  int scaledWidth = outputWidth * renderScale;
  int scaledHeight = outputWidth * renderScale;
  
  size_t outputPixelArraySize =
      scaledWidth * scaledHeight * 4 * sizeof(unsigned char);
  unsigned char *d_outputPixelArray;
  unsigned char *h_outputPixelArray = new unsigned char[outputPixelArraySize];

  size_t staticBodiesArraySize = numStaticBodies * sizeof(kernel::StaticBody);
  kernel::StaticBody *d_staticBodies;

  cudaMalloc((void **)&d_outputPixelArray, outputPixelArraySize);
  cudaMalloc((void **)&d_staticBodies, staticBodiesArraySize);

  cudaMemcpy(d_staticBodies, h_staticBodies, staticBodiesArraySize,
             cudaMemcpyHostToDevice);

  dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE_Z);
  dim3 gridSize((scaledWidth + blockSize.x - 1) / blockSize.x,
                (scaledHeight + blockSize.y - 1) / blockSize.y);

  kernel::
      renderBasins<<<gridSize, blockSize,
                     BLOCK_SIZE * BLOCK_SIZE * sizeof(kernel::particleData)>>>(
          d_outputPixelArray, scaledWidth, scaledHeight, d_staticBodies,
          numStaticBodies, renderScale);

  // Wait for the kernel to complete
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    // Handle error
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }

  cudaMemcpy(h_outputPixelArray, d_outputPixelArray, outputPixelArraySize,
             cudaMemcpyDeviceToHost);

  cudaFree(d_outputPixelArray);

  return h_outputPixelArray;
}

} // namespace kernel