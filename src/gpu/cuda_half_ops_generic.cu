#include <mpi.h>

extern "C" {
  #define IMPORT_FROM_EXTERN_C
  #include "main.h"
  #undef IMPORT_FROM_EXTERN_C
  #include "operator.h"
}

#include "cuda_miscellaneous.h"
#include "cuda_half.h"
#include "cuda_dirac_kernels_componentwise_PRECISION.h"
#include "cuda_half_ops_PRECISION.h"

#ifdef CUDA_OPT

extern "C" void cuda_half_ops_PRECISION_init( operator_PRECISION_struct *op ) {
  for ( int mu=0; mu<4; mu++ ) {
    op->Ds_half_gpu[mu] = NULL;
    op->Ds_half_scale[mu] = 1.0f;
  }
  op->clover_half_gpu = NULL;
  op->clover_half_scale = 1.0f;
  op->half_valid = 0;
}

extern "C" void cuda_half_ops_PRECISION_invalidate( operator_PRECISION_struct *op ) {
  op->half_valid = 0;
}

extern "C" void cuda_half_ops_PRECISION_free( operator_PRECISION_struct *op, level_struct *l ) {
  const size_t n = l->num_inner_lattice_sites;
  const unsigned int css = clover_site_size( l->num_lattice_site_var, l->depth );
  for ( int mu=0; mu<4; mu++ ) {
    if ( op->Ds_half_gpu[mu] != NULL ) CUDA_FREE( op->Ds_half_gpu[mu], __half2, 9*n );
  }
  if ( op->clover_half_gpu != NULL ) CUDA_FREE( op->clover_half_gpu, __half2, css*n );
  op->half_valid = 0;
}

extern "C" int cuda_half_ops_PRECISION( operator_PRECISION_struct *op, level_struct *l ) {
#ifdef CUDA_HALF_OPERATORS
  if ( sizeof(PRECISION) != sizeof(float) || l->depth != 0 || !g.cuda_half_solve || !g.cuda_half_smoother ) return 0;
  if ( op->half_valid ) return 1;

  // build the half-precision copies from the float arrays (same componentwise layout)
  const size_t n = l->num_inner_lattice_sites;
  const unsigned int css = clover_site_size( l->num_lattice_site_var, l->depth );
  for ( int mu=0; mu<4; mu++ ) {
    if ( op->Ds_half_gpu[mu] == NULL ) CUDA_MALLOC( op->Ds_half_gpu[mu], __half2, 9*n );
    op->Ds_half_scale[mu] = cuda_complex_float_to_half( (__half2*) op->Ds_half_gpu[mu],
                                                        (cuFloatComplex const*) op->Ds_componentwise_gpu[mu], 9*n );
  }
  if ( g.csw != 0 ) {
    if ( op->clover_half_gpu == NULL ) CUDA_MALLOC( op->clover_half_gpu, __half2, css*n );
    op->clover_half_scale = cuda_complex_float_to_half( (__half2*) op->clover_half_gpu,
                                                        (cuFloatComplex const*) op->clover_componentwise_gpu, css*n );
  }
  op->half_valid = 1;
  return 1;
#else
  return 0;
#endif
}

// the three self-coupling launchers: same launch configuration as in cuda_dirac_generic.cu
#define CUDA_OP_CLOVER_LAUNCH( kernel ) do { \
  constexpr size_t blockSize = 128; \
  const size_t gridSize = minGridSizeForN( num_sites, blockSize ); \
  const unsigned int css = clover_site_size( l->num_lattice_site_var, l->depth ); \
  if ( cuda_half_ops_PRECISION( op, l ) && op->clover_half_gpu != NULL ) { \
    kernel##_half_PRECISION<<<gridSize, blockSize>>>( eta, phi, ((__half2 const*) op->clover_half_gpu) + css*start, \
                                                      op->clover_half_scale, num_sites ); \
  } else { \
    kernel##_PRECISION<<<gridSize, blockSize>>>( eta, phi, op->clover_componentwise_gpu + css*start, num_sites ); \
  } \
  cuda_safe_call( cudaDeviceSynchronize() ); } while(0)

void cuda_op_clover_PRECISION( cuda_vector_PRECISION eta, cuda_vector_PRECISION phi,
                               operator_PRECISION_struct *op, size_t start, int num_sites,
                               level_struct *l ) {
  CUDA_OP_CLOVER_LAUNCH( cuda_site_clover_componentwise );
}

void cuda_op_diag_ee_PRECISION( cuda_vector_PRECISION eta, cuda_vector_PRECISION phi,
                                operator_PRECISION_struct *op, size_t start, int num_sites,
                                level_struct *l ) {
  CUDA_OP_CLOVER_LAUNCH( cuda_site_diag_ee_componentwise );
}

void cuda_op_diag_oo_inv_PRECISION( cuda_vector_PRECISION eta, cuda_vector_PRECISION phi,
                                    operator_PRECISION_struct *op, size_t start, int num_sites,
                                    level_struct *l ) {
  CUDA_OP_CLOVER_LAUNCH( cuda_site_diag_oo_inv_componentwise );
}

#ifdef CUDA_HALF_TEST
// relative difference || a - b || / || b || of two GPU vectors of length n (all ranks)
static double cuda_half_test_reldiff_PRECISION( cu_cmplx_PRECISION const *a_gpu, cu_cmplx_PRECISION const *b_gpu,
                                                size_t n ) {
  cu_cmplx_PRECISION *a = (cu_cmplx_PRECISION*) malloc( n*sizeof(cu_cmplx_PRECISION) );
  cu_cmplx_PRECISION *b = (cu_cmplx_PRECISION*) malloc( n*sizeof(cu_cmplx_PRECISION) );
  cuda_safe_call( cudaMemcpy( a, a_gpu, n*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost ) );
  cuda_safe_call( cudaMemcpy( b, b_gpu, n*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost ) );
  double loc[2] = {0, 0}, glob[2];
  for ( size_t i=0; i<n; i++ ) {
    double dr = (double)a[i].x - (double)b[i].x, di = (double)a[i].y - (double)b[i].y;
    loc[0] += dr*dr + di*di;
    loc[1] += (double)b[i].x*b[i].x + (double)b[i].y*b[i].y;
  }
  MPI_Allreduce( loc, glob, 2, MPI_DOUBLE, MPI_SUM, g.comm_cart );
  free( a ); free( b );
  return glob[1] > 0 ? sqrt( glob[0]/glob[1] ) : sqrt( glob[0] );
}

// out = op(in) with the float operator data and with the half-precision copies; prints the difference
#define CUDA_HALF_TEST_COMPARE( name, apply ) do { \
  cuda_safe_call( cudaMemset( y_f, 0, n*sizeof(cu_cmplx_PRECISION) ) ); \
  cuda_safe_call( cudaMemset( y_h, 0, n*sizeof(cu_cmplx_PRECISION) ) ); \
  g.cuda_half_solve = 0; { cu_cmplx_PRECISION *y = y_f; apply; } \
  g.cuda_half_solve = 1; { cu_cmplx_PRECISION *y = y_h; apply; } \
  cuda_safe_call( cudaDeviceSynchronize() ); \
  printf0( "CUDA_HALF_TEST: %-28s || half - float || / || float || = %e\n", name, \
           cuda_half_test_reldiff_PRECISION( y_h, y_f, n ) ); } while(0)

extern "C" void cuda_half_ops_PRECISION_test( level_struct *l ) {
  if ( sizeof(PRECISION) != sizeof(float) || l->depth != 0 ) return;
  const int save = g.cuda_half_solve, save_sm = g.cuda_half_smoother;
  g.cuda_half_smoother = 1;
  const size_t n = l->inner_vector_size;
  cu_cmplx_PRECISION *x = NULL, *y_f = NULL, *y_h = NULL;
  cuda_safe_call( cudaMalloc( (void**) &x, n*sizeof(cu_cmplx_PRECISION) ) );
  cuda_safe_call( cudaMalloc( (void**) &y_f, n*sizeof(cu_cmplx_PRECISION) ) );
  cuda_safe_call( cudaMalloc( (void**) &y_h, n*sizeof(cu_cmplx_PRECISION) ) );
  {
    cu_cmplx_PRECISION *x_host = (cu_cmplx_PRECISION*) malloc( n*sizeof(cu_cmplx_PRECISION) );
    srand( 1234 + g.my_rank );
    for ( size_t i=0; i<n; i++ ) {
      x_host[i] = make_cu_cmplx_PRECISION( (PRECISION)( (double)rand()/RAND_MAX - 0.5 ), (PRECISION)( (double)rand()/RAND_MAX - 0.5 ) );
    }
    cuda_safe_call( cudaMemcpy( x, x_host, n*sizeof(cu_cmplx_PRECISION), cudaMemcpyHostToDevice ) );
    free( x_host );
  }

  operator_PRECISION_struct *oe_op = &(l->oe_op_PRECISION);
  if ( oe_op->clover_componentwise_gpu != NULL && oe_op->Ds_componentwise_gpu[0] != NULL ) {
    const size_t start_odd = (size_t) oe_op->num_even_sites * l->num_lattice_site_var;
    CUDA_HALF_TEST_COMPARE( "odd-even D_oo^-1", 
      cuda_op_diag_oo_inv_PRECISION( y+start_odd, x+start_odd, oe_op, oe_op->num_even_sites, oe_op->num_odd_sites, l ) );
    CUDA_HALF_TEST_COMPARE( "odd-even D_ee",
      cuda_op_diag_ee_PRECISION( y, x, oe_op, 0, oe_op->num_even_sites, l ) );
    CUDA_HALF_TEST_COMPARE( "odd-even hopping D_eo",
      cuda_hopping_term_PRECISION( y, x, oe_op, _EVEN_SITES, l ) );
    CUDA_HALF_TEST_COMPARE( "Schur complement",
      cuda_apply_schur_complement_PRECISION( y, x, oe_op, l ) );
  }

  g.cuda_half_solve = save;
  g.cuda_half_smoother = save_sm;
  cuda_safe_call( cudaFree( x ) );
  cuda_safe_call( cudaFree( y_f ) );
  cuda_safe_call( cudaFree( y_h ) );
}
#endif

#endif
