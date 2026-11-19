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
//
// The kernels use one thread per row of a matrix-vector product: the threads of a site read the
// same entries of the vector and consecutive entries of each column of the matrix.

static constexpr int coarse_oe_block_size = 128;


// i-th row of A*x, A in the vectorized layout
__device__ static __forceinline__ cu_cmplx_PRECISION
cuda_coarse_vectorized_row_PRECISION( const OPERATOR_TYPE_PRECISION *A, int lda,
                                      const cu_cmplx_PRECISION *x, int n, int i ) {
  PRECISION re = 0, im = 0;
  for ( int j=0; j<n; j++ ) {
    PRECISION a_re = A[2*j*lda+i], a_im = A[(2*j+1)*lda+i];
    cu_cmplx_PRECISION xj = x[j];
    re += a_re*xj.x - a_im*xj.y;
    im += a_re*xj.y + a_im*xj.x;
  }
  return make_cu_cmplx_PRECISION( re, im );
}


// y_s = A_s x_s on the sites s = start..start+num_sites-1 (as coarse_self_couplings_PRECISION_vectorized)
__global__ void cuda_coarse_self_couplings_kernel_PRECISION( cu_cmplx_PRECISION *y, const cu_cmplx_PRECISION *x,
                                                             const OPERATOR_TYPE_PRECISION *clover,
                                                             int start, int num_sites, int n, int lda ) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if ( idx >= num_sites*n ) return;
  int s = start + idx/n, i = idx%n;
  y[s*n+i] = cuda_coarse_vectorized_row_PRECISION( clover + (size_t)s*2*n*lda, lda, x + (size_t)s*n, n, i );
}


// backward couplings of the sites start..start+num_sites-1 whose forward neighbor is a ghost site:
// out_{x+mu} += factor * Dt_mu(x) in_x (first part of coarse_pn_hopping_term_PRECISION_vectorized)
__global__ void cuda_coarse_hopping_boundary_kernel_PRECISION( cu_cmplx_PRECISION *out, const cu_cmplx_PRECISION *in,
                                                               const OPERATOR_TYPE_PRECISION *Dt, const int *fw,
                                                               int start, int num_sites, int num_inner_sites,
                                                               int n, int lda, PRECISION factor ) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if ( idx >= num_sites*4*n ) return;
  int k = start + idx/(4*n), mu = (idx/n)%4, i = idx%n;
  int nb = fw[5*k+1+mu];
  if ( nb < num_inner_sites ) return;
  int s = fw[5*k];
  size_t link_offset = (size_t)2*n*lda;
  cu_cmplx_PRECISION v = cuda_coarse_vectorized_row_PRECISION( Dt + (4*(size_t)s+mu)*link_offset, lda,
                                                               in + (size_t)s*n, n, i );
  cu_cmplx_PRECISION *o = out + (size_t)nb*n + i;
  o->x += factor*v.x;
  o->y += factor*v.y;
}


// out_x += factor * ( sum_mu Dt_mu(x-mu) in_{x-mu} [x-mu inner site] + sum_mu D_mu(x) in_{x+mu} )
// on the sites start..start+num_sites-1 (second part of coarse_pn_hopping_term_PRECISION_vectorized)
__global__ void cuda_coarse_hopping_kernel_PRECISION( cu_cmplx_PRECISION *out, const cu_cmplx_PRECISION *in,
                                                      const OPERATOR_TYPE_PRECISION *D,
                                                      const OPERATOR_TYPE_PRECISION *Dt,
                                                      const int *fw, const int *bw,
                                                      int start, int num_sites, int num_inner_sites,
                                                      int n, int lda, PRECISION factor ) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if ( idx >= num_sites*n ) return;
  int k = start + idx/n, i = idx%n;
  int s = fw[5*k];
  size_t link_offset = (size_t)2*n*lda;
  PRECISION re = 0, im = 0;
  for ( int mu=0; mu<4; mu++ ) {
    int b = bw[5*k+1+mu];
    // the terms from ghost sites are added by cuda_ghost_wait_PRECISION
    if ( b >= num_inner_sites ) continue;
    cu_cmplx_PRECISION v = cuda_coarse_vectorized_row_PRECISION( Dt + (4*(size_t)b+mu)*link_offset, lda,
                                                                 in + (size_t)b*n, n, i );
    re += v.x; im += v.y;
  }
  for ( int mu=0; mu<4; mu++ ) {
    cu_cmplx_PRECISION v = cuda_coarse_vectorized_row_PRECISION( D + (4*(size_t)s+mu)*link_offset, lda,
                                                                 in + (size_t)fw[5*k+1+mu]*n, n, i );
    re += v.x; im += v.y;
  }
  cu_cmplx_PRECISION *o = out + (size_t)s*n + i;
  o->x += factor*re;
  o->y += factor*im;
}


static inline int cuda_coarse_oddeven_lda_PRECISION( level_struct *l ) {
#ifdef VECTORIZE_COARSE_OPERATOR_PRECISION
  return SIMD_LENGTH_PRECISION*((l->num_lattice_site_var+SIMD_LENGTH_PRECISION-1)/SIMD_LENGTH_PRECISION);
#else
  return l->num_lattice_site_var;
#endif
}


static inline unsigned int cuda_coarse_oddeven_grid_PRECISION( size_t num_threads ) {
  return (unsigned int)( (num_threads + coarse_oe_block_size - 1)/coarse_oe_block_size );
}


extern "C" int cuda_coarse_operator_on_gpu_PRECISION( operator_PRECISION_struct *op, level_struct *l ) {
#ifdef VECTORIZE_COARSE_OPERATOR_PRECISION
  return ( l->depth > 0 && l->level > 0 && op->D_vectorized_gpu != NULL );
#else
  return 0;
#endif
}


extern "C" int cuda_coarse_oddeven_on_gpu_PRECISION( level_struct *l ) {
  return cuda_coarse_operator_on_gpu_PRECISION( &(l->oe_op_PRECISION), l );
}


// GPU copy of a coarse operator in the vectorized layout (l->oe_op_PRECISION or l->s_PRECISION.op):
// couplings of the inner sites, neighbor tables, ghost exchange as in op->c, four vectors
extern "C" void cuda_coarse_operator_PRECISION_alloc( operator_PRECISION_struct *op, level_struct *l ) {
#ifdef VECTORIZE_COARSE_OPERATOR_PRECISION
  if ( l->depth == 0 || l->level == 0 ) return;

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

  // communication: same ghost layout as op->c
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
    CUDA_MALLOC( op->cuda_c.boundary_table_gpu[2*mu], int, bs );
    cuda_safe_call( cudaMemcpy( op->cuda_c.boundary_table_gpu[2*mu], op->c.boundary_table[2*mu], bs*sizeof(int),
                                cudaMemcpyHostToDevice ) );
    if ( op->c.boundary_table[2*mu+1] == op->c.boundary_table[2*mu] ) {
      // same table in both directions (odd-even operator)
      op->cuda_c.boundary_table_gpu[2*mu+1] = op->cuda_c.boundary_table_gpu[2*mu];
    } else {
      CUDA_MALLOC( op->cuda_c.boundary_table_gpu[2*mu+1], int, bs );
      cuda_safe_call( cudaMemcpy( op->cuda_c.boundary_table_gpu[2*mu+1], op->c.boundary_table[2*mu+1], bs*sizeof(int),
                                  cudaMemcpyHostToDevice ) );
    }
  }

  cuda_coarse_operator_PRECISION_op_to_gpu( op, l );
#endif
}


extern "C" void cuda_coarse_operator_PRECISION_op_to_gpu( operator_PRECISION_struct *op, level_struct *l ) {
#ifdef VECTORIZE_COARSE_OPERATOR_PRECISION
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


extern "C" void cuda_coarse_operator_PRECISION_free( operator_PRECISION_struct *op, level_struct *l ) {
#ifdef VECTORIZE_COARSE_OPERATOR_PRECISION
  // only the operators allocated by cuda_coarse_operator_PRECISION_alloc
  if ( l->depth == 0 || l->level == 0 || op->D_vectorized_gpu == NULL ) return;

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
    if ( op->cuda_c.boundary_table_gpu[2*mu+1] != op->cuda_c.boundary_table_gpu[2*mu] )
      CUDA_FREE( op->cuda_c.boundary_table_gpu[2*mu+1], int, bs );
    op->cuda_c.boundary_table_gpu[2*mu+1] = NULL;
    CUDA_FREE( op->cuda_c.boundary_table_gpu[2*mu], int, bs );
  }
#endif
}


extern "C" void cuda_coarse_oddeven_PRECISION_alloc( level_struct *l ) {
  cuda_coarse_operator_PRECISION_alloc( &(l->oe_op_PRECISION), l );
}


extern "C" void cuda_coarse_oddeven_PRECISION_op_to_gpu( level_struct *l ) {
  cuda_coarse_operator_PRECISION_op_to_gpu( &(l->oe_op_PRECISION), l );
}


extern "C" void cuda_coarse_oddeven_PRECISION_free( level_struct *l ) {
  cuda_coarse_operator_PRECISION_free( &(l->oe_op_PRECISION), l );
}


// y = A x on the sites start..end-1, with A = clover_vectorized: the self coupling on the even sites
// (coarse_diag_ee_PRECISION) and its inverse on the odd sites (coarse_diag_oo_inv_PRECISION)
extern "C" void cuda_coarse_self_couplings_PRECISION( cuda_vector_PRECISION y, cuda_vector_PRECISION x,
                                                      int start, int end, operator_PRECISION_struct *op,
                                                      level_struct *l ) {
  int n = l->num_lattice_site_var, num_sites = end-start;
  if ( num_sites <= 0 ) return;
  cuda_coarse_self_couplings_kernel_PRECISION<<< cuda_coarse_oddeven_grid_PRECISION( (size_t)num_sites*n ),
                                                 coarse_oe_block_size >>>
    ( y, x, op->clover_vectorized_gpu, start, num_sites, n, cuda_coarse_oddeven_lda_PRECISION( l ) );
  cuda_safe_call( cudaGetLastError() );
  cuda_safe_call( cudaDeviceSynchronize() );
}


// as coarse_pn_hopping_term_PRECISION_vectorized: sign = +1 is coarse_hopping_term_PRECISION
// (out -= hopping term), sign = -1 is coarse_n_hopping_term_PRECISION (out += hopping term);
// amount = _EVEN_SITES or _ODD_SITES are the sites of out that are computed (odd-even operator),
// _FULL_SYSTEM all the sites (coarse_hopping_term_PRECISION( ..., _FULL_SYSTEM, ... ), any operator)
extern "C" void cuda_coarse_pn_hopping_term_PRECISION( cuda_vector_PRECISION out, cuda_vector_PRECISION in,
                                                       operator_PRECISION_struct *op, const int amount, int sign,
                                                       level_struct *l ) {
  int n = l->num_lattice_site_var, lda = cuda_coarse_oddeven_lda_PRECISION( l ),
      num_inner_sites = l->num_inner_lattice_sites, start, num_sites,
      plus_dir_param = _FULL_SYSTEM, minus_dir_param = _FULL_SYSTEM;
  PRECISION factor = ( sign == +1 ) ? -1.0 : 1.0;
  int comm = ( l->num_processes > 1 && op->c.comm );

  if ( amount == _EVEN_SITES ) {
    minus_dir_param = _ODD_SITES;
    plus_dir_param = _EVEN_SITES;
  } else if ( amount == _ODD_SITES ) {
    minus_dir_param = _EVEN_SITES;
    plus_dir_param = _ODD_SITES;
  }

  if ( comm ) {
    // ghost sites of out = 0
    cuda_safe_call( cudaMemset( out + l->inner_vector_size, 0,
                                (l->vector_size-l->inner_vector_size)*sizeof(cu_cmplx_PRECISION) ) );
    for ( int mu=0; mu<4; mu++ )
      cuda_ghost_sendrecv_PRECISION( in, mu, -1, &(op->cuda_c), minus_dir_param, l );

    // backward couplings into the forward ghost sites of out, from the sites of the other parity
    if ( amount == _EVEN_SITES ) {
      start = op->num_even_sites; num_sites = op->num_odd_sites;
    } else if ( amount == _ODD_SITES ) {
      start = 0; num_sites = op->num_even_sites;
    } else {
      start = 0; num_sites = num_inner_sites;
    }
    if ( num_sites > 0 ) {
      cuda_coarse_hopping_boundary_kernel_PRECISION<<< cuda_coarse_oddeven_grid_PRECISION( (size_t)num_sites*4*n ),
                                                       coarse_oe_block_size >>>
        ( out, in, op->D_transformed_vectorized_gpu, op->neighbor_fw_gpu, start, num_sites, num_inner_sites,
          n, lda, factor );
      cuda_safe_call( cudaGetLastError() );
    }
    // the ghost sites of out are sent right away (MPI_Isend from the GPU buffer)
    cuda_safe_call( cudaDeviceSynchronize() );

    for ( int mu=0; mu<4; mu++ )
      cuda_ghost_sendrecv_PRECISION( out, mu, +1, &(op->cuda_c), plus_dir_param, l );
    for ( int mu=0; mu<4; mu++ )
      cuda_ghost_wait_PRECISION( in, mu, -1, &(op->cuda_c), minus_dir_param, l );
  }

  if ( amount == _EVEN_SITES ) {
    start = 0; num_sites = op->num_even_sites;
  } else if ( amount == _ODD_SITES ) {
    start = op->num_even_sites; num_sites = op->num_odd_sites;
  } else {
    start = 0; num_sites = num_inner_sites;
  }
  if ( num_sites > 0 ) {
    cuda_coarse_hopping_kernel_PRECISION<<< cuda_coarse_oddeven_grid_PRECISION( (size_t)num_sites*n ),
                                            coarse_oe_block_size >>>
      ( out, in, op->D_vectorized_gpu, op->D_transformed_vectorized_gpu, op->neighbor_fw_gpu, op->neighbor_bw_gpu,
        start, num_sites, num_inner_sites, n, lda, factor );
    cuda_safe_call( cudaGetLastError() );
  }
  cuda_safe_call( cudaDeviceSynchronize() );

  if ( comm ) {
    // adds the backward couplings computed by the neighbors (cuda_ghost_wait_PRECISION, depth > 0)
    for ( int mu=0; mu<4; mu++ )
      cuda_ghost_wait_PRECISION( out, mu, +1, &(op->cuda_c), plus_dir_param, l );
  }
}


// out_e = ( D_ee - D_eo D_oo^{-1} D_oe ) in_e, as coarse_apply_schur_complement_PRECISION
// (only the even sites of in are read and only the even sites of out are written)
extern "C" void cuda_coarse_apply_schur_complement_PRECISION( cuda_vector_PRECISION out, cuda_vector_PRECISION in,
                                                              operator_PRECISION_struct *op, level_struct *l ) {
  cuda_vector_PRECISION tmp0 = op->buffer_gpu[0], tmp1 = op->buffer_gpu[1];
  int n = l->num_lattice_site_var, n_even = op->num_even_sites, n_odd = op->num_odd_sites;

  // out_e = D_ee in_e
  cuda_coarse_self_couplings_PRECISION( out, in, 0, n_even, op, l );
  // tmp0_o = - D_oe in_e
  cuda_safe_call( cudaMemset( tmp0 + (size_t)n_even*n, 0, (size_t)n_odd*n*sizeof(cu_cmplx_PRECISION) ) );
  cuda_coarse_pn_hopping_term_PRECISION( tmp0, in, op, _ODD_SITES, +1, l );
  // tmp1_o = D_oo^{-1} tmp0_o
  cuda_coarse_self_couplings_PRECISION( tmp1, tmp0, n_even, n_even+n_odd, op, l );
  // out_e += D_eo tmp1_o
  cuda_coarse_pn_hopping_term_PRECISION( out, tmp1, op, _EVEN_SITES, -1, l );
}


extern "C" void cuda_coarse_apply_schur_complement_PRECISION_vectorwrapper( vector_PRECISION out, vector_PRECISION in,
                                                                            operator_PRECISION_struct *op,
                                                                            level_struct *l,
                                                                            struct Thread *threading ) {
  size_t even_size = (size_t)op->num_even_sites*l->num_lattice_site_var;

  SYNC_CORES(threading)
  START_MASTER(threading)
  cuda_safe_call( cudaMemcpy( op->coarse_in_gpu, in, even_size*sizeof(cu_cmplx_PRECISION), cudaMemcpyHostToDevice ) );
  cuda_coarse_apply_schur_complement_PRECISION( op->coarse_out_gpu, op->coarse_in_gpu, op, l );
  cuda_safe_call( cudaMemcpy( out, op->coarse_out_gpu, even_size*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost ) );
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)
}

// eta = D phi on all the inner sites, as apply_coarse_operator_PRECISION (vectorized layout)
extern "C" void cuda_apply_coarse_operator_PRECISION( cuda_vector_PRECISION eta, cuda_vector_PRECISION phi,
                                                      operator_PRECISION_struct *op, level_struct *l ) {
  cuda_coarse_self_couplings_PRECISION( eta, phi, 0, l->num_inner_lattice_sites, op, l );
  cuda_coarse_pn_hopping_term_PRECISION( eta, phi, op, _FULL_SYSTEM, +1, l );
}


extern "C" void cuda_apply_coarse_operator_PRECISION_vectorwrapper( vector_PRECISION eta, vector_PRECISION phi,
                                                                    operator_PRECISION_struct *op, level_struct *l,
                                                                    struct Thread *threading ) {
  size_t size = (size_t)l->inner_vector_size;

  SYNC_CORES(threading)
  START_MASTER(threading)
  cuda_safe_call( cudaMemcpy( op->coarse_in_gpu, phi, size*sizeof(cu_cmplx_PRECISION), cudaMemcpyHostToDevice ) );
  cuda_apply_coarse_operator_PRECISION( op->coarse_out_gpu, op->coarse_in_gpu, op, l );
  cuda_safe_call( cudaMemcpy( eta, op->coarse_out_gpu, size*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost ) );
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)
}

#endif
