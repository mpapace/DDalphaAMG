#include <mpi.h>

extern "C"{

  #define IMPORT_FROM_EXTERN_C
  #include "main.h"
  #undef IMPORT_FROM_EXTERN_C

}

#include "cuda_ghost_PRECISION.h"

#ifdef CUDA_OPT

// Odd-even operator of a coarse level (depth > 0) on the GPU.
//
// The data are copies of the ones used by the CPU code with VECTORIZE_COARSE_OPERATOR_PRECISION
// (see coarse_oddeven_setup_PRECISION_set_couplings): per site, D_vectorized holds the four links
// U_mu(x), D_transformed_vectorized the four links of the backward couplings, and clover_vectorized
// the self coupling on the even sites and its inverse on the odd sites. Each one is a dense
// n x n complex matrix (n = l->num_lattice_site_var), stored column by column: the real parts of
// column j at A[2*j*lda + i] and its imaginary parts at A[(2*j+1)*lda + i], i = 0..n-1, with
// lda = n rounded up to a multiple of SIMD_LENGTH_PRECISION (see cgemv and cgenmv in sse_blas_vectorized.h).


static inline int cuda_coarse_oddeven_lda_PRECISION( level_struct *l ) {
#ifdef VECTORIZE_COARSE_OPERATOR_PRECISION
  return SIMD_LENGTH_PRECISION*((l->num_lattice_site_var+SIMD_LENGTH_PRECISION-1)/SIMD_LENGTH_PRECISION);
#else
  return l->num_lattice_site_var;
#endif
}


extern "C" int cuda_coarse_oddeven_on_gpu_PRECISION( level_struct *l ) {
#ifdef VECTORIZE_COARSE_OPERATOR_PRECISION
  return ( l->depth > 0 && l->level > 0 && l->oe_op_PRECISION.D_vectorized_gpu != NULL );
#else
  return 0;
#endif
}


extern "C" void cuda_coarse_oddeven_PRECISION_alloc( level_struct *l ) {
#ifdef VECTORIZE_COARSE_OPERATOR_PRECISION
  if ( l->depth == 0 || l->level == 0 ) return;

  operator_PRECISION_struct *op = &(l->oe_op_PRECISION);
  int n = l->num_inner_lattice_sites, lda = cuda_coarse_oddeven_lda_PRECISION( l );
  size_t offset_v = (size_t)2*l->num_lattice_site_var*lda;

  CUDA_MALLOC( op->D_vectorized_gpu, OPERATOR_TYPE_PRECISION, 4*offset_v*n );
  CUDA_MALLOC( op->D_transformed_vectorized_gpu, OPERATOR_TYPE_PRECISION, 4*offset_v*n );
  CUDA_MALLOC( op->clover_vectorized_gpu, OPERATOR_TYPE_PRECISION, offset_v*n );
  CUDA_MALLOC( op->neighbor_fw_gpu, int, 5*n );
  CUDA_MALLOC( op->neighbor_bw_gpu, int, 5*n );
  CUDA_MALLOC( op->coarse_in_gpu, cu_cmplx_PRECISION, l->vector_size );
  CUDA_MALLOC( op->coarse_out_gpu, cu_cmplx_PRECISION, l->vector_size );
  CUDA_MALLOC( op->buffer_gpu[0], cu_cmplx_PRECISION, l->vector_size );
  CUDA_MALLOC( op->buffer_gpu[1], cu_cmplx_PRECISION, l->vector_size );

  // neighbor tables (fixed)
  cuda_safe_call( cudaMemcpy( op->neighbor_fw_gpu, op->neighbor_table, 5*n*sizeof(int), cudaMemcpyHostToDevice ) );
  cuda_safe_call( cudaMemcpy( op->neighbor_bw_gpu, op->backward_neighbor_table, 5*n*sizeof(int), cudaMemcpyHostToDevice ) );

  // communication: same ghost layout as op->c (see coarse_oddeven_setup_PRECISION)
  cuda_ghost_alloc_PRECISION( 0, &(op->cuda_c), l );
  for ( int i=0; i<8; i++ ) {
    op->cuda_c.comm_start[i] = op->c.comm_start[i];
    op->cuda_c.num_boundary_sites[i] = op->c.num_boundary_sites[i];
    op->cuda_c.num_even_boundary_sites[i] = op->c.num_even_boundary_sites[i];
    op->cuda_c.num_odd_boundary_sites[i] = op->c.num_odd_boundary_sites[i];
  }
  op->cuda_c.comm = op->c.comm;
  for ( int mu=0; mu<4; mu++ ) {
    int bs = 1;
    for ( int nu=0; nu<4; nu++ )
      if ( nu != mu ) bs *= l->local_lattice[nu];
    // as for op->c, the same table in both directions
    CUDA_MALLOC( op->cuda_c.boundary_table_gpu[2*mu], int, bs );
    cuda_safe_call( cudaMemcpy( op->cuda_c.boundary_table_gpu[2*mu], op->c.boundary_table[2*mu], bs*sizeof(int),
                                cudaMemcpyHostToDevice ) );
    op->cuda_c.boundary_table_gpu[2*mu+1] = op->cuda_c.boundary_table_gpu[2*mu];
  }

  cuda_coarse_oddeven_PRECISION_op_to_gpu( l );
#endif
}


extern "C" void cuda_coarse_oddeven_PRECISION_op_to_gpu( level_struct *l ) {
#ifdef VECTORIZE_COARSE_OPERATOR_PRECISION
  operator_PRECISION_struct *op = &(l->oe_op_PRECISION);
  if ( op->D_vectorized_gpu == NULL ) return;

  int n = l->num_inner_lattice_sites, lda = cuda_coarse_oddeven_lda_PRECISION( l );
  size_t offset_v = (size_t)2*l->num_lattice_site_var*lda;

  cuda_safe_call( cudaMemcpy( op->D_vectorized_gpu, op->D_vectorized, 4*offset_v*n*sizeof(OPERATOR_TYPE_PRECISION),
                              cudaMemcpyHostToDevice ) );
  cuda_safe_call( cudaMemcpy( op->D_transformed_vectorized_gpu, op->D_transformed_vectorized,
                              4*offset_v*n*sizeof(OPERATOR_TYPE_PRECISION), cudaMemcpyHostToDevice ) );
  cuda_safe_call( cudaMemcpy( op->clover_vectorized_gpu, op->clover_vectorized, offset_v*n*sizeof(OPERATOR_TYPE_PRECISION),
                              cudaMemcpyHostToDevice ) );
#endif
}


extern "C" void cuda_coarse_oddeven_PRECISION_free( level_struct *l ) {
#ifdef VECTORIZE_COARSE_OPERATOR_PRECISION
  operator_PRECISION_struct *op = &(l->oe_op_PRECISION);
  if ( op->D_vectorized_gpu == NULL ) return;

  int n = l->num_inner_lattice_sites, lda = cuda_coarse_oddeven_lda_PRECISION( l );
  size_t offset_v = (size_t)2*l->num_lattice_site_var*lda;

  CUDA_FREE( op->D_vectorized_gpu, OPERATOR_TYPE_PRECISION, 4*offset_v*n );
  CUDA_FREE( op->D_transformed_vectorized_gpu, OPERATOR_TYPE_PRECISION, 4*offset_v*n );
  CUDA_FREE( op->clover_vectorized_gpu, OPERATOR_TYPE_PRECISION, offset_v*n );
  CUDA_FREE( op->neighbor_fw_gpu, int, 5*n );
  CUDA_FREE( op->neighbor_bw_gpu, int, 5*n );
  CUDA_FREE( op->coarse_in_gpu, cu_cmplx_PRECISION, l->vector_size );
  CUDA_FREE( op->coarse_out_gpu, cu_cmplx_PRECISION, l->vector_size );
  CUDA_FREE( op->buffer_gpu[0], cu_cmplx_PRECISION, l->vector_size );
  CUDA_FREE( op->buffer_gpu[1], cu_cmplx_PRECISION, l->vector_size );

  cuda_ghost_free_PRECISION( &(op->cuda_c), l );
  for ( int mu=0; mu<4; mu++ ) {
    int bs = 1;
    for ( int nu=0; nu<4; nu++ )
      if ( nu != mu ) bs *= l->local_lattice[nu];
    CUDA_FREE( op->cuda_c.boundary_table_gpu[2*mu], int, bs );
    op->cuda_c.boundary_table_gpu[2*mu+1] = NULL;
  }
#endif
}

#endif
