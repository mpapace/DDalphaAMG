#ifndef CUDA_INTERPOLATION_PRECISION_HEADER
#define CUDA_INTERPOLATION_PRECISION_HEADER

#ifdef CUDA_OPT
  void cuda_interpolation_PRECISION_op_to_gpu( level_struct *l );
  void cuda_interpolation_PRECISION_free( level_struct *l );
  void cuda_restrict_PRECISION( vector_PRECISION phi_c, vector_PRECISION phi, level_struct *l, struct Thread *threading );
  void cuda_interpolate_PRECISION( vector_PRECISION phi, vector_PRECISION phi_c, level_struct *l, struct Thread *threading, int add );
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
