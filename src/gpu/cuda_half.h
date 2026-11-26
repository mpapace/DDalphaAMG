/** \file cuda_half.h
 *
 *  \brief Half-precision storage of operator data on the GPU (build with -DCUDA_HALF_OPERATORS).
 *
 *  Operator arrays (gauge links, clover term, ...) can be kept in IEEE half precision (__half2 for
 *  a complex number) with one scale per array: value = scale * (stored half value). The kernels
 *  read them through HalfComponentAccess, which converts to the compute type on the fly, so all
 *  arithmetic stays in single precision.
 */

#ifndef CUDA_HALF_H
#define CUDA_HALF_H

#include <cuda_fp16.h>
#include <cuComplex.h>
#include "cuda_componentwise.h"

template <typename T> __device__ inline T cu_cmplx_from_floats(float re, float im);
template <> __device__ inline cuFloatComplex cu_cmplx_from_floats<cuFloatComplex>(float re, float im) {
  return make_cuFloatComplex(re, im);
}
template <> __device__ inline cuDoubleComplex cu_cmplx_from_floats<cuDoubleComplex>(float re, float im) {
  return make_cuDoubleComplex((double)re, (double)im);
}

/** Read-only counterpart of ComponentAccess for half-precision data with one scale per array:
 *  HalfComponentAccess<T>(data, num_sites, scale)[i] == scale * data[i * num_sites], as T. */
template <typename T>
class HalfComponentAccess {
 public:
  __host__ __device__ HalfComponentAccess(__half2 const* data, size_t num_sites, float scale)
      : data(data), num_sites(num_sites), scale(scale) {}
  __device__ T operator[](size_t i) const {
    float2 v = __half22float2(data[i * num_sites]);
    return cu_cmplx_from_floats<T>(scale * v.x, scale * v.y);
  }

 private:
  __half2 const* data;
  size_t num_sites;
  float scale;
};

/** Componentwise access to an operator array stored in T (the scale is ignored) ... */
template <typename T>
__device__ inline ComponentAccess<T const> make_op_access(T const* data, size_t num_sites, float scale) {
  return ComponentAccess<T const>(data, num_sites);
}
/** ... or in half precision. */
template <typename T>
__device__ inline HalfComponentAccess<T> make_op_access(__half2 const* data, size_t num_sites, float scale) {
  return HalfComponentAccess<T>(data, num_sites, scale);
}

/** Converts n single-precision complex numbers to half precision on the GPU: dst = src / scale
 *  with scale = max over all real and imaginary parts of |src| (1 if they are all zero).
 *  Returns the scale. dst must hold n __half2. */
float cuda_complex_float_to_half(__half2* dst, cuFloatComplex const* src, size_t n);

#endif  // CUDA_HALF_H
