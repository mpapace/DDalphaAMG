#ifndef CUDA_INTERPOLATION_PRECISION_HEADER
#define CUDA_INTERPOLATION_PRECISION_HEADER

#ifdef CUDA_OPT
  // data of the interpolation on the GPU (called by the proxies in proxies/interpolation_proxy_generic.c)
  void cuda_interpolation_PRECISION_alloc( level_struct *l );
  void cuda_interpolation_PRECISION_op_to_gpu( level_struct *l );
  void cuda_interpolation_PRECISION_free( level_struct *l );

  // on GPU vectors (one thread, default stream, not synchronised)
  void cuda_restrict_PRECISION( cuda_vector_PRECISION phi_c_gpu, cuda_vector_PRECISION phi_gpu, level_struct *l );
  void cuda_interpolate_PRECISION( cuda_vector_PRECISION phi_gpu, cuda_vector_PRECISION phi_c_gpu, level_struct *l, int add );

  // on host vectors (called by all threads, through the proxies)
  void cuda_restrict_PRECISION_vectorwrapper( vector_PRECISION phi_c, vector_PRECISION phi, level_struct *l, struct Thread *threading );
  void cuda_interpolate_PRECISION_vectorwrapper( vector_PRECISION phi, vector_PRECISION phi_c, level_struct *l, struct Thread *threading );
  void cuda_interpolate3_PRECISION_vectorwrapper( vector_PRECISION phi, vector_PRECISION phi_c, level_struct *l, struct Thread *threading );

  // depth-0 smoother on the GPU (called by vcycle_PRECISION)
  int  cuda_fine_residual_PRECISION( vector_PRECISION r, vector_PRECISION phi, vector_PRECISION eta,
                                     operator_PRECISION_struct *op, level_struct *l, struct Thread *threading,
                                     int keep_on_gpu );
  void cuda_smoother_oe_prepare_PRECISION( vector_PRECISION b, level_struct *l, struct Thread *threading );
  void cuda_smoother_oe_finish_PRECISION( vector_PRECISION phi, vector_PRECISION x, level_struct *l,
                                          struct Thread *threading );
  int  cuda_smoother_oe_gmres_PRECISION( vector_PRECISION phi, gmres_PRECISION_struct *p, level_struct *l,
                                         struct Thread *threading );
#endif

#endif
