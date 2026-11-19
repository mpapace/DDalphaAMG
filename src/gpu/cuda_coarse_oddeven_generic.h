#ifndef CUDA_COARSE_ODDEVEN_PRECISION_HEADER
#define CUDA_COARSE_ODDEVEN_PRECISION_HEADER

#ifdef CUDA_OPT
  // odd-even operator of a coarse level on the GPU (l->oe_op_PRECISION, depth > 0, not the coarsest level),
  // in the vectorized layout of D_vectorized, D_transformed_vectorized and clover_vectorized
  // (VECTORIZE_COARSE_OPERATOR_PRECISION, i.e. SSE builds in single precision)

  // 1 if the odd-even operator of this level is applied on the GPU
  int  cuda_coarse_oddeven_on_gpu_PRECISION( level_struct *l );

  // data on the GPU (called by coarse_oddeven_setup_PRECISION, coarse_oddeven_re_setup_PRECISION
  // and coarse_oddeven_free_PRECISION, one thread)
  void cuda_coarse_oddeven_PRECISION_alloc( level_struct *l );
  void cuda_coarse_oddeven_PRECISION_op_to_gpu( level_struct *l );
  void cuda_coarse_oddeven_PRECISION_free( level_struct *l );
#endif

#endif
