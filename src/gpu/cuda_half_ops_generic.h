/** \file cuda_half_ops_generic.h
 *
 *  \brief Fine-level operators in half precision on the GPU (-DCUDA_HALF_OPERATORS).
 *
 *  Inside the depth-0 smoother (g.cuda_half_smoother) during the solve phase (g.cuda_half_solve)
 *  the gauge links of a depth-0
 *  single-precision operator are read from half-precision copies (one scale per array) built
 *  lazily from the float arrays at their first use. Arithmetic stays in single precision.
 *  Without CUDA_HALF_OPERATORS, outside the smoother or the solve phase, in double precision or at depth > 0 the
 *  float arrays are used. The hopping term (cuda_hopping_term, cuda_d_plus_clover) asks cuda_half_ops_PRECISION which
 *  storage to use.
 */

#ifndef CUDA_HALF_OPS_PRECISION_H
#define CUDA_HALF_OPS_PRECISION_H

#include "global_enums.h"

/** Returns 1 if op is to be applied from its half-precision copies, building them if needed. */
extern "C" int cuda_half_ops_PRECISION( operator_PRECISION_struct *op, level_struct *l );
/** Initializes the half-precision fields of op (no allocation). */
extern "C" void cuda_half_ops_PRECISION_init( operator_PRECISION_struct *op );
/** Marks the half-precision copies of op as outdated (call after writing the float arrays). */
extern "C" void cuda_half_ops_PRECISION_invalidate( operator_PRECISION_struct *op );
/** Frees the half-precision copies of op, if any. */
extern "C" void cuda_half_ops_PRECISION_free( operator_PRECISION_struct *op, level_struct *l );

/** With -DCUDA_HALF_TEST: prints the relative difference between half and float storage for the
 *  odd-even hopping term and the Schur complement. Master only. */
extern "C" void cuda_half_ops_PRECISION_test( level_struct *l );

#endif  // CUDA_HALF_OPS_PRECISION_H
