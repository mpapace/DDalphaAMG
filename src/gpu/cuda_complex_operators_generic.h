/**
 * \file cuda_complex_operators_generic.h
 * 
 * C++ operator overloads for CUDA complex numbers.
 */


#ifndef CUDA_COMPLEX_OPERATORS_PRECISION_H
#define CUDA_COMPLEX_OPERATORS_PRECISION_H

#include "cuda_complex.h"

/**
 * \brief Create a literal which is of type cu_cmplx_precision.
 * 
 * This only yields an imaginary part. Thus a term like '1.0_cu_i_PRECISION' yields a complex
 * number 0.0 + 1.0i, which is processable on the GPU (i.e. of type 'cu_cmplx_PRECISION'). Both
 * float and double is supported.
 * 
 * This method is not to be called directly but provides an operator as per C++ specification.
 * 
 * The method is a constexpr and thus capable to populate variables with static storage duration.
 * 
 * \param imag    The literal to process. All conversion is automatic and the type 'long double'
 *                should only be taken in a symbolic sense and is part of the processing that occurs
 *                when writing a literal like '1.0_cu_i_PRECISION'.
 * \return        A complex number with an imaginary part of value 'imag'.
 */
__host__ __device__ constexpr cu_cmplx_PRECISION operator ""_cu_i_PRECISION(long double imag) {
  return {0, (PRECISION)imag};
}

// Defined inline here (not in a .cu file): with separate compilation (-rdc) out-of-line
// operators cannot be inlined into the kernels of other files, so every complex product or sum
// became a function call with arguments passed by reference, forcing the operands of the kernels
// into local memory.
__host__ __device__ inline cu_cmplx_PRECISION operator*(cu_cmplx_PRECISION const& lhs, int const& rhs) {
  return cu_cmul_PRECISION(lhs, make_cu_cmplx_PRECISION(rhs, 0));
}

__host__ __device__ inline cu_cmplx_PRECISION operator*(int const& lhs, cu_cmplx_PRECISION const& rhs) {
  return cu_cmul_PRECISION(make_cu_cmplx_PRECISION(lhs, 0), rhs);
}

__host__ __device__ inline cu_cmplx_PRECISION operator*(cu_cmplx_PRECISION const& lhs,
                                                 cu_cmplx_PRECISION const& rhs) {
  return cu_cmul_PRECISION(lhs, rhs);
}

__host__ __device__ inline cu_cmplx_PRECISION operator-(cu_cmplx_PRECISION const& lhs,
                                                 cu_cmplx_PRECISION const& rhs) {
  return cu_csub_PRECISION(lhs, rhs);
}

__host__ __device__ inline cu_cmplx_PRECISION operator+(cu_cmplx_PRECISION const& lhs,
                                                 cu_cmplx_PRECISION const& rhs) {
  return cu_cadd_PRECISION(lhs, rhs);
}

__host__ __device__ inline cu_cmplx_PRECISION & operator+=(cu_cmplx_PRECISION & lhs,
                                                    cu_cmplx_PRECISION const& rhs) {
  lhs = cu_cadd_PRECISION(lhs, rhs);
  return lhs;
}

__host__ __device__ inline cu_cmplx_PRECISION & operator-=(cu_cmplx_PRECISION & lhs,
                                                    cu_cmplx_PRECISION const& rhs) {
  lhs = cu_csub_PRECISION(lhs, rhs);
  return lhs;
}

__host__ __device__ inline cu_cmplx_PRECISION operator-(cu_cmplx_PRECISION const& value) {
  return {-value.x, -value.y};
}

#endif  // CUDA_COMPLEX_OPERATORS_PRECISION_H
