#ifndef CUDA_COARSE_ODDEVEN_PRECISION_HEADER
#define CUDA_COARSE_ODDEVEN_PRECISION_HEADER

#ifdef CUDA_OPT
  // operators of a coarse level on the GPU (depth > 0, not the coarsest level): the odd-even operator
  // l->oe_op_PRECISION and the operator l->s_PRECISION.op, in the vectorized layout of D_vectorized,
  // D_transformed_vectorized and clover_vectorized (VECTORIZE_COARSE_OPERATOR_PRECISION, i.e. SSE builds
  // in single precision)

  // 1 if the operator (odd-even operator of this level) is applied on the GPU
  int  cuda_coarse_operator_on_gpu_PRECISION( operator_PRECISION_struct *op, level_struct *l );
  int  cuda_coarse_oddeven_on_gpu_PRECISION( level_struct *l );

  // data of a coarse operator in the vectorized layout on the GPU (one thread)
  void cuda_coarse_operator_PRECISION_alloc( operator_PRECISION_struct *op, level_struct *l );
  void cuda_coarse_operator_PRECISION_op_to_gpu( operator_PRECISION_struct *op, level_struct *l );
  void cuda_coarse_operator_PRECISION_free( operator_PRECISION_struct *op, level_struct *l );

  // data on the GPU (called by coarse_oddeven_setup_PRECISION, coarse_oddeven_re_setup_PRECISION
  // and coarse_oddeven_free_PRECISION, one thread)
  void cuda_coarse_oddeven_PRECISION_alloc( level_struct *l );
  void cuda_coarse_oddeven_PRECISION_op_to_gpu( level_struct *l );
  void cuda_coarse_oddeven_PRECISION_free( level_struct *l );

  // on GPU vectors of length l->vector_size (one thread, default stream, synchronised on return)
  void cuda_coarse_self_couplings_PRECISION( cuda_vector_PRECISION y, cuda_vector_PRECISION x,
                                             int start, int end, operator_PRECISION_struct *op, level_struct *l );
  void cuda_coarse_pn_hopping_term_PRECISION( cuda_vector_PRECISION out, cuda_vector_PRECISION in,
                                              operator_PRECISION_struct *op, const int amount, int sign,
                                              level_struct *l );
  void cuda_coarse_apply_schur_complement_PRECISION( cuda_vector_PRECISION out, cuda_vector_PRECISION in,
                                                     operator_PRECISION_struct *op, level_struct *l );
  void cuda_apply_coarse_operator_PRECISION( cuda_vector_PRECISION eta, cuda_vector_PRECISION phi,
                                             operator_PRECISION_struct *op, level_struct *l );
  void cuda_coarse_solve_odd_even_prepare_PRECISION( cuda_vector_PRECISION x, cuda_vector_PRECISION b,
                                                     operator_PRECISION_struct *op, level_struct *l );
  void cuda_coarse_solve_odd_even_finish_PRECISION( cuda_vector_PRECISION x, cuda_vector_PRECISION b,
                                                    operator_PRECISION_struct *op, level_struct *l );

  // on host vectors (called by all threads)
  void cuda_coarse_apply_schur_complement_PRECISION_vectorwrapper( vector_PRECISION out, vector_PRECISION in,
                                                                   operator_PRECISION_struct *op, level_struct *l,
                                                                   struct Thread *threading );
  // first and last step of coarse_solve_odd_even_PRECISION (around its GMRES)
  void cuda_coarse_solve_odd_even_prepare_PRECISION_vectorwrapper( vector_PRECISION x, vector_PRECISION b,
                                                                   operator_PRECISION_struct *op, level_struct *l,
                                                                   struct Thread *threading );
  void cuda_coarse_solve_odd_even_finish_PRECISION_vectorwrapper( vector_PRECISION x, vector_PRECISION b,
                                                                  operator_PRECISION_struct *op, level_struct *l,
                                                                  struct Thread *threading );
  // coarse_solve_odd_even_PRECISION (smoother of the coarse level) with the vectors on the GPU,
  // returns the number of GMRES iterations
  int  cuda_coarse_solve_odd_even_PRECISION( gmres_PRECISION_struct *p, operator_PRECISION_struct *op,
                                             level_struct *l, struct Thread *threading );
  void cuda_apply_coarse_operator_PRECISION_vectorwrapper( vector_PRECISION eta, vector_PRECISION phi,
                                                           operator_PRECISION_struct *op, level_struct *l,
                                                           struct Thread *threading );
#endif

#endif
