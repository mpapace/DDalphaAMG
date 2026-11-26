#include <cuda_runtime.h>
#include <math.h>
#include "cuda_half.h"
#include <stdio.h>
#include <string.h>
#include "miscellaneous.h"

// max of |x| over the real and imaginary parts; non-negative floats compare like their bit patterns
__global__ void cuda_half_absmax_kernel(unsigned int* out, cuFloatComplex const* src, size_t n) {
  float m = 0.0f;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    m = fmaxf(m, fmaxf(fabsf(src[i].x), fabsf(src[i].y)));
  }
  for (int off = 16; off > 0; off /= 2) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, off));
  if ((threadIdx.x & 31) == 0) atomicMax(out, __float_as_uint(m));
}

__global__ void cuda_half_convert_kernel(__half2* dst, cuFloatComplex const* src, size_t n, float inv_scale) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    dst[i] = __floats2half2_rn(inv_scale * src[i].x, inv_scale * src[i].y);
  }
}

float cuda_complex_float_to_half(__half2* dst, cuFloatComplex const* src, size_t n) {
  unsigned int* d_max = NULL;
  unsigned int h_max = 0;
  cuda_safe_call(cudaMalloc((void**)&d_max, sizeof(unsigned int)));
  cuda_safe_call(cudaMemset(d_max, 0, sizeof(unsigned int)));
  const int block = 256;
  int grid = (int)((n + block - 1) / block);
  if (grid > 4096) grid = 4096;
  if (grid < 1) grid = 1;
  cuda_half_absmax_kernel<<<grid, block>>>(d_max, src, n);
  cuda_safe_call(cudaMemcpy(&h_max, d_max, sizeof(unsigned int), cudaMemcpyDeviceToHost));
  cuda_safe_call(cudaFree(d_max));
  float scale;
  memcpy(&scale, &h_max, sizeof(float));
  if (!(scale > 0.0f)) scale = 1.0f;
  cuda_half_convert_kernel<<<grid, block>>>(dst, src, n, 1.0f / scale);
  cuda_safe_call(cudaDeviceSynchronize());
  cuda_safe_call(cudaGetLastError());
  return scale;
}
