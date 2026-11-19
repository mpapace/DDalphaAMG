#include <mpi.h>
#include <cublas_v2.h>
#include <complex>
#include <vector>

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
  if ( op->gmres_V_gpu != NULL ) {
    CUDA_FREE( op->gmres_V_gpu, cu_cmplx_PRECISION, (size_t)op->gmres_V_cols*l->vector_size );
    CUDA_FREE( op->gmres_b_gpu, cu_cmplx_PRECISION, l->vector_size );
    CUDA_FREE( op->gmres_x_gpu, cu_cmplx_PRECISION, l->vector_size );
    CUDA_FREE( op->gmres_r_gpu, cu_cmplx_PRECISION, l->vector_size );
    CUDA_FREE( op->gmres_h_gpu, cu_cmplx_PRECISION, op->gmres_V_cols );
    op->gmres_V_cols = 0;
  }

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

// first step of coarse_solve_odd_even_PRECISION on GPU vectors (one thread):
// x_o = D_oo^{-1} b_o, then b_e = b_e + (coarse_n_hopping_term) x_o
extern "C" void cuda_coarse_solve_odd_even_prepare_PRECISION( cuda_vector_PRECISION x, cuda_vector_PRECISION b,
                                                              operator_PRECISION_struct *op, level_struct *l ) {
  cuda_coarse_self_couplings_PRECISION( x, b, op->num_even_sites, op->num_even_sites+op->num_odd_sites, op, l );
  cuda_coarse_pn_hopping_term_PRECISION( b, x, op, _EVEN_SITES, -1, l );
}


// last step of coarse_solve_odd_even_PRECISION on GPU vectors (one thread):
// b_o = b_o + (coarse_n_hopping_term) x_e, then x_o = D_oo^{-1} b_o
extern "C" void cuda_coarse_solve_odd_even_finish_PRECISION( cuda_vector_PRECISION x, cuda_vector_PRECISION b,
                                                             operator_PRECISION_struct *op, level_struct *l ) {
  cuda_coarse_pn_hopping_term_PRECISION( b, x, op, _ODD_SITES, -1, l );
  cuda_coarse_self_couplings_PRECISION( x, b, op->num_even_sites, op->num_even_sites+op->num_odd_sites, op, l );
}


// the two steps above on host vectors (called by all threads); only the parts of x and b that
// are read or written are copied
extern "C" void cuda_coarse_solve_odd_even_prepare_PRECISION_vectorwrapper( vector_PRECISION x, vector_PRECISION b,
                                                                            operator_PRECISION_struct *op,
                                                                            level_struct *l, struct Thread *threading ) {
  size_t even_size = (size_t)op->num_even_sites*l->num_lattice_site_var,
         odd_size = (size_t)op->num_odd_sites*l->num_lattice_site_var;
  cuda_vector_PRECISION x_gpu = op->coarse_in_gpu, b_gpu = op->coarse_out_gpu;

  SYNC_CORES(threading)
  START_MASTER(threading)
  cuda_safe_call( cudaMemcpy( b_gpu, b, (even_size+odd_size)*sizeof(cu_cmplx_PRECISION), cudaMemcpyHostToDevice ) );
  cuda_coarse_solve_odd_even_prepare_PRECISION( x_gpu, b_gpu, op, l );
  cuda_safe_call( cudaMemcpy( x+even_size, x_gpu+even_size, odd_size*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost ) );
  cuda_safe_call( cudaMemcpy( b, b_gpu, even_size*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost ) );
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)
}


extern "C" void cuda_coarse_solve_odd_even_finish_PRECISION_vectorwrapper( vector_PRECISION x, vector_PRECISION b,
                                                                           operator_PRECISION_struct *op,
                                                                           level_struct *l, struct Thread *threading ) {
  size_t even_size = (size_t)op->num_even_sites*l->num_lattice_site_var,
         odd_size = (size_t)op->num_odd_sites*l->num_lattice_site_var;
  cuda_vector_PRECISION x_gpu = op->coarse_in_gpu, b_gpu = op->coarse_out_gpu;

  SYNC_CORES(threading)
  START_MASTER(threading)
  cuda_safe_call( cudaMemcpy( x_gpu, x, even_size*sizeof(cu_cmplx_PRECISION), cudaMemcpyHostToDevice ) );
  cuda_safe_call( cudaMemcpy( b_gpu+even_size, b+even_size, odd_size*sizeof(cu_cmplx_PRECISION), cudaMemcpyHostToDevice ) );
  cuda_coarse_solve_odd_even_finish_PRECISION( x_gpu, b_gpu, op, l );
  cuda_safe_call( cudaMemcpy( b+even_size, b_gpu+even_size, odd_size*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost ) );
  cuda_safe_call( cudaMemcpy( x+even_size, x_gpu+even_size, odd_size*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost ) );
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)
}

#define CUBLAS_CHECK_COARSE(call)                                                 \
  do {                                                                            \
    cublasStatus_t st_ = (call);                                                  \
    if ( st_ != CUBLAS_STATUS_SUCCESS ) {                                         \
      fprintf( stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__, __LINE__ ); \
      MPI_Abort( MPI_COMM_WORLD, 1 );                                             \
    }                                                                             \
  } while (0)

// cuBLAS calls for both precisions (overloads, so that this generic file works for float and double)
static inline cublasStatus_t coarse_blas_gemv( cublasHandle_t h, cublasOperation_t t, int m, int n,
                                               const cuFloatComplex *alpha, const cuFloatComplex *A, int lda,
                                               const cuFloatComplex *x, const cuFloatComplex *beta, cuFloatComplex *y ) {
  return cublasCgemv( h, t, m, n, alpha, A, lda, x, 1, beta, y, 1 ); }
static inline cublasStatus_t coarse_blas_gemv( cublasHandle_t h, cublasOperation_t t, int m, int n,
                                               const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
                                               const cuDoubleComplex *x, const cuDoubleComplex *beta, cuDoubleComplex *y ) {
  return cublasZgemv( h, t, m, n, alpha, A, lda, x, 1, beta, y, 1 ); }
static inline cublasStatus_t coarse_blas_nrm2( cublasHandle_t h, int n, const cuFloatComplex *x, float *r ) {
  return cublasScnrm2( h, n, x, 1, r ); }
static inline cublasStatus_t coarse_blas_nrm2( cublasHandle_t h, int n, const cuDoubleComplex *x, double *r ) {
  return cublasDznrm2( h, n, x, 1, r ); }
static inline cublasStatus_t coarse_blas_rscal( cublasHandle_t h, int n, float a, cuFloatComplex *x ) {
  return cublasCsscal( h, n, &a, x, 1 ); }
static inline cublasStatus_t coarse_blas_rscal( cublasHandle_t h, int n, double a, cuDoubleComplex *x ) {
  return cublasZdscal( h, n, &a, x, 1 ); }
static inline cublasStatus_t coarse_blas_copy( cublasHandle_t h, int n, const cuFloatComplex *x, cuFloatComplex *y ) {
  return cublasCcopy( h, n, x, 1, y, 1 ); }
static inline cublasStatus_t coarse_blas_copy( cublasHandle_t h, int n, const cuDoubleComplex *x, cuDoubleComplex *y ) {
  return cublasZcopy( h, n, x, 1, y, 1 ); }
static inline cublasStatus_t coarse_blas_axpy( cublasHandle_t h, int n, const cuFloatComplex *a,
                                               const cuFloatComplex *x, cuFloatComplex *y ) {
  return cublasCaxpy( h, n, a, x, 1, y, 1 ); }
static inline cublasStatus_t coarse_blas_axpy( cublasHandle_t h, int n, const cuDoubleComplex *a,
                                               const cuDoubleComplex *x, cuDoubleComplex *y ) {
  return cublasZaxpy( h, n, a, x, 1, y, 1 ); }

static cublasHandle_t coarse_cublas_PRECISION = NULL;


// GMRES on the Schur complement of the odd-even operator of a coarse level, entirely on the GPU:
// b and x are the even sites of GPU vectors of length l->vector_size. Same algorithm as
// fgmres_PRECISION for the depth > 0 smoother: no preconditioner, classical Gram-Schmidt (inner
// products with v_0..v_j, then the norm of w), restart_length steps per restart, num_restart
// restarts, stop if the estimated relative residual drops below p->tol. Inner products and norms
// are reduced over the processes of the level on the host (MPI_Allreduce, as on the CPU); the small
// Hessenberg problem is solved on the host. The basis vectors have ghost sites (the Schur complement
// receives them), so their stride is l->vector_size.
static int cuda_coarse_gmres_PRECISION( gmres_PRECISION_struct *p, operator_PRECISION_struct *op, level_struct *l,
                                        cu_cmplx_PRECISION *x, cu_cmplx_PRECISION *b ) {

  typedef std::complex<double> cd;
  const int m = p->restart_length;
  const int n = op->num_even_sites * l->num_lattice_site_var;
  const int ld = l->vector_size;
  MPI_Comm comm = l->gs_PRECISION.level_comm;

  if ( coarse_cublas_PRECISION == NULL ) {
    CUBLAS_CHECK_COARSE( cublasCreate( &coarse_cublas_PRECISION ) );
    CUBLAS_CHECK_COARSE( cublasSetPointerMode( coarse_cublas_PRECISION, CUBLAS_POINTER_MODE_HOST ) );
  }
  cublasHandle_t h = coarse_cublas_PRECISION;
  if ( op->gmres_V_cols < m+1 ) {
    if ( op->gmres_V_gpu != NULL ) {
      CUDA_FREE( op->gmres_V_gpu, cu_cmplx_PRECISION, (size_t)op->gmres_V_cols*ld );
      CUDA_FREE( op->gmres_h_gpu, cu_cmplx_PRECISION, op->gmres_V_cols );
    }
    CUDA_MALLOC( op->gmres_V_gpu, cu_cmplx_PRECISION, (size_t)(m+1)*ld );
    CUDA_MALLOC( op->gmres_h_gpu, cu_cmplx_PRECISION, m+1 );
    op->gmres_V_cols = m+1;
  }

  cu_cmplx_PRECISION *V = op->gmres_V_gpu, *r = op->gmres_r_gpu, *hd = op->gmres_h_gpu;
  const cu_cmplx_PRECISION one = make_cu_cmplx_PRECISION(1,0), zero = make_cu_cmplx_PRECISION(0,0),
                           mone = make_cu_cmplx_PRECISION(-1,0);
  std::vector<cd> H( (size_t)m*(m+1) ), gam( m+1 ), sn( m ), cs( m ), y( m );
  std::vector<cu_cmplx_PRECISION> hloc( m+1 ), hglob( m+1 );
  auto Hc = [&]( int col, int row ) -> cd& { return H[ (size_t)col*(m+1) + row ]; };   // H[j][i] of the CPU code

  // global 2-norm over the even sites of all the processes of the level
  auto global_norm = [&]( const cu_cmplx_PRECISION *v ) -> double {
    PRECISION nl = 0;
    CUBLAS_CHECK_COARSE( coarse_blas_nrm2( h, n, v, &nl ) );
    PRECISION sq = nl*nl, sq_glob = sq;
    if ( g.num_processes > 1 )
      MPI_Allreduce( &sq, &sq_glob, 1, MPI_PRECISION, MPI_SUM, comm );
    return sqrt( (double)sq_glob );
  };

  double norm_r0 = 1, gamma_jp1 = 1;
  int finish = 0, iter = 0, j = -1;

  for ( int ol=0; ol<p->num_restart && finish==0; ol++ ) {
    int res;
    if ( ol == 0 && p->initial_guess_zero ) {
      res = _NO_RES;
      CUBLAS_CHECK_COARSE( coarse_blas_copy( h, n, b, r ) );                    // r = b
    } else {
      res = _RES;
      cuda_coarse_apply_schur_complement_PRECISION( r, x, op, l );             // r = S x
      CUBLAS_CHECK_COARSE( coarse_blas_rscal( h, n, (PRECISION)(-1.0), r ) );  // r = b - S x
      CUBLAS_CHECK_COARSE( coarse_blas_axpy( h, n, &one, b, r ) );
    }
    double gamma0 = global_norm( r );
    gam[0] = gamma0;
    if ( ol == 0 ) norm_r0 = gamma0;
    CUBLAS_CHECK_COARSE( coarse_blas_copy( h, n, r, V ) );                      // v_0 = r / gamma_0
    CUBLAS_CHECK_COARSE( coarse_blas_rscal( h, n, (PRECISION)(1.0/gamma0), V ) );

    for ( int il=0; il<m && finish==0; il++ ) {
      j = il; iter++;
      cu_cmplx_PRECISION *vj = V + (size_t)j*ld, *w = V + (size_t)(j+1)*ld;    // w is built in place of v_{j+1}

      cuda_coarse_apply_schur_complement_PRECISION( w, vj, op, l );            // w = S v_j

      // h_i = <v_i, w>, i = 0..j, reduced over the processes
      CUBLAS_CHECK_COARSE( coarse_blas_gemv( h, CUBLAS_OP_C, n, j+1, &one, V, ld, w, &zero, hd ) );
      cuda_safe_call( cudaMemcpy( hloc.data(), hd, (j+1)*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost ) );
      if ( g.num_processes > 1 )
        MPI_Allreduce( hloc.data(), hglob.data(), j+1, MPI_COMPLEX_PRECISION, MPI_SUM, comm );
      else
        hglob = hloc;
      for ( int i=0; i<=j; i++ ) Hc(j,i) = cd( hglob[i].x, hglob[i].y );

      // w = w - sum_i h_i v_i  (classical Gram-Schmidt, as on the CPU)
      cuda_safe_call( cudaMemcpy( hd, hglob.data(), (j+1)*sizeof(cu_cmplx_PRECISION), cudaMemcpyHostToDevice ) );
      CUBLAS_CHECK_COARSE( coarse_blas_gemv( h, CUBLAS_OP_N, n, j+1, &mone, V, ld, hd, &one, w ) );

      // h_{j+1} = ||w||, v_{j+1} = w / h_{j+1}
      double hn = global_norm( w );
      Hc(j,j+1) = hn;
      if ( hn > 1e-15 ) CUBLAS_CHECK_COARSE( coarse_blas_rscal( h, n, (PRECISION)(1.0/hn), w ) );

      if ( std::abs( Hc(j,j+1) ) > p->tol/10 ) {
        // Givens rotations (qr_update_PRECISION)
        for ( int i=0; i<j; i++ ) {
          cd beta = (-sn[i])*Hc(j,i) + cs[i]*Hc(j,i+1);
          Hc(j,i) = std::conj(cs[i])*Hc(j,i) + std::conj(sn[i])*Hc(j,i+1);
          Hc(j,i+1) = beta;
        }
        double beta = sqrt( std::norm( Hc(j,j) ) + std::norm( Hc(j,j+1) ) );
        sn[j] = Hc(j,j+1)/beta; cs[j] = Hc(j,j)/beta;
        gam[j+1] = (-sn[j])*gam[j]; gam[j] = std::conj(cs[j])*gam[j];
        Hc(j,j) = beta; Hc(j,j+1) = 0;
        gamma_jp1 = std::abs( gam[j+1] );
        if ( gamma_jp1/norm_r0 < p->tol || gamma_jp1/norm_r0 > 1E+5 ) {
          finish = 1;
          if ( gamma_jp1/norm_r0 > 1E+5 ) printf0("Divergence of cuda_coarse_gmres_PRECISION, iter = %d\n", iter );
        }
      } else {
        printf0("depth: %d, iter: %d, coarse smoother GMRES (GPU) H(%d,%d) = %+lf+%lfi\n", l->depth, iter, j+1, j,
                Hc(j,j+1).real(), Hc(j,j+1).imag() );
        finish = 1;
        break;
      }
    }

    // x = x + sum_i y_i v_i (or x = sum_i y_i v_i after a zero initial guess): compute_solution_PRECISION
    for ( int i=j; i>=0; i-- ) {
      y[i] = gam[i];
      for ( int k=i+1; k<=j; k++ ) y[i] -= Hc(k,i)*y[k];
      y[i] /= Hc(i,i);
    }
    for ( int i=0; i<=j; i++ ) hglob[i] = make_cu_cmplx_PRECISION( (PRECISION)y[i].real(), (PRECISION)y[i].imag() );
    cuda_safe_call( cudaMemcpy( hd, hglob.data(), (j+1)*sizeof(cu_cmplx_PRECISION), cudaMemcpyHostToDevice ) );
    int add = ( res == _NO_RES ) ? ol : 1;
    CUBLAS_CHECK_COARSE( coarse_blas_gemv( h, CUBLAS_OP_N, n, j+1, &one, V, ld, hd, add ? &one : &zero, x ) );
  }
  cuda_safe_call( cudaDeviceSynchronize() );
  return iter;
}


// coarse_solve_odd_even_PRECISION with the vectors on the GPU (called by all threads): copy of b
// (and of x if the initial guess is not zero) to the GPU, first step, GMRES on the Schur complement,
// last step, copy of x to the host. p->b is not updated on the host (the callers only use p->x).
// Returns the number of GMRES iterations.
extern "C" int cuda_coarse_solve_odd_even_PRECISION( gmres_PRECISION_struct *p, operator_PRECISION_struct *op,
                                                     level_struct *l, struct Thread *threading ) {
  int iter = 0;
  size_t inner_size = (size_t)l->inner_vector_size,
         even_size = (size_t)op->num_even_sites*l->num_lattice_site_var;

  SYNC_CORES(threading)
  START_MASTER(threading)
  if ( op->gmres_b_gpu == NULL ) {
    CUDA_MALLOC( op->gmres_b_gpu, cu_cmplx_PRECISION, l->vector_size );
    CUDA_MALLOC( op->gmres_x_gpu, cu_cmplx_PRECISION, l->vector_size );
    CUDA_MALLOC( op->gmres_r_gpu, cu_cmplx_PRECISION, l->vector_size );
  }
  cu_cmplx_PRECISION *x = op->gmres_x_gpu, *b = op->gmres_b_gpu;
  cuda_safe_call( cudaMemcpy( b, p->b, inner_size*sizeof(cu_cmplx_PRECISION), cudaMemcpyHostToDevice ) );
  if ( !p->initial_guess_zero )
    cuda_safe_call( cudaMemcpy( x, p->x, even_size*sizeof(cu_cmplx_PRECISION), cudaMemcpyHostToDevice ) );
  cuda_coarse_solve_odd_even_prepare_PRECISION( x, b, op, l );
  iter = cuda_coarse_gmres_PRECISION( p, op, l, x, b );
  cuda_coarse_solve_odd_even_finish_PRECISION( x, b, op, l );
  cuda_safe_call( cudaMemcpy( p->x, x, inner_size*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost ) );
  ((int*)threading->workspace)[0] = iter;
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)
  return ((int*)threading->workspace)[0];
}

#endif
