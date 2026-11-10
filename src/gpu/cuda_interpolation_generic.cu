#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cublas_v2.h>
#include <complex>
#include <vector>

extern "C"{

  #define IMPORT_FROM_EXTERN_C
  #include "main.h"
  #undef IMPORT_FROM_EXTERN_C

  #include "operator.h"
}

#include "cuda_componentwise.h"

#ifdef CUDA_OPT

#define CUDA_CHECK_INTERP(call)                                                   \
  do {                                                                            \
    cudaError_t err_ = (call);                                                    \
    if ( err_ != cudaSuccess ) {                                                  \
      fprintf( stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,           \
               cudaGetErrorString(err_) );                                        \
      MPI_Abort( MPI_COMM_WORLD, 1 );                                             \
    }                                                                             \
  } while (0)


// memory layout of l->is_PRECISION.op in this build
#if defined( SSE ) && defined( INTERPOLATION_OPERATOR_LAYOUT_OPTIMIZED_PRECISION )
static constexpr bool restrict_sse_layout_PRECISION = true;
#else
static constexpr bool restrict_sse_layout_PRECISION = false;
#endif


// SIMD block length of the SSE layout (unused for the plain layout)
static int restrict_simd_length_PRECISION( void ) {
#ifdef SSE
  return SIMD_LENGTH_PRECISION;
#else
  return 1;
#endif
}


// number of complex entries of l->is_PRECISION.op, depending on its memory layout
static size_t interpolation_op_size_PRECISION( level_struct *l ) {
#if defined( SSE ) && defined( INTERPOLATION_OPERATOR_LAYOUT_OPTIMIZED_PRECISION )
  return (size_t)OPERATOR_COMPONENT_OFFSET_PRECISION * (size_t)l->vector_size;
#else
  return (size_t)l->num_eig_vect * (size_t)l->inner_vector_size;
#endif
}


// reads entry (row, j) of the interpolation operator, where
//   row = ((i*agg_sites + k)*2 + k1)*npev + k2   (same index as the fine vector phi)
//   j   = test vector index
template<bool SSE_LAYOUT>
__device__ __forceinline__ void restrict_op_entry_PRECISION( const cu_cmplx_PRECISION *op, size_t row, int j, int nev,
                                                             size_t vector_size, int S, PRECISION *re, PRECISION *im ) {
  if ( SSE_LAYOUT ) {
    // SSE layout: test vectors in blocks of S, block jb starts at jb*S*vector_size;
    // each row stores [re_0 .. re_{S-1}, im_0 .. im_{S-1}]
    const PRECISION *opf = (const PRECISION*) op;
    size_t base = 2*( (size_t)(j/S)*S*vector_size + (size_t)S*row );
    *re = opf[ base + j%S ];
    *im = opf[ base + S + j%S ];
  } else {
    // plain layout (interpolation_generic.c): interleaved complex numbers, test vector index fastest
    cu_cmplx_PRECISION o = op[ row*nev + j ];
    *re = cu_creal_PRECISION(o);
    *im = cu_cimag_PRECISION(o);
  }
}


// One CUDA block per (aggregate i, spin half k1): blockIdx.x = 2*i + k1 (= output offset / nev).
// threadIdx.x = test vector j (nev padded to a multiple of 32), threadIdx.y = slice of the rows of the aggregate:
// each thread sums rows y, y+Y, y+2Y, ...; the Y partial sums are then reduced in shared memory
// with a fixed tree order (deterministic result).
template<bool SSE_LAYOUT>
__global__ void cuda_restrict_PRECISION_kernel( cuda_vector_PRECISION phi_c, const cu_cmplx_PRECISION *phi,
                                                const cu_cmplx_PRECISION *op, int nev, int npev,
                                                int agg_sites, size_t vector_size, int S ) {

  extern __shared__ __align__(16) unsigned char restrict_smem_PRECISION[];
  PRECISION *red_re = (PRECISION*) restrict_smem_PRECISION;
  PRECISION *red_im = red_re + blockDim.x*blockDim.y;

  const int j = threadIdx.x, y = threadIdx.y, X = blockDim.x, Y = blockDim.y;
  const int i = blockIdx.x / 2, k1 = blockIdx.x % 2;

  PRECISION re = 0, im = 0;

  if ( j < nev ) {
    const int n_rows = agg_sites*npev;
    // row = ((i*agg_sites + k)*2 + k1)*npev + k2 = row0 + k*2*npev + k2
    const size_t row0 = (size_t)i*agg_sites*2*npev + (size_t)k1*npev;
    #pragma unroll 4
    for ( int r=y; r<n_rows; r+=Y ) {
      int k = r / npev, k2 = r - k*npev;
      size_t row = row0 + (size_t)k*2*npev + k2;
      cu_cmplx_PRECISION p = phi[row];
      PRECISION p_re = cu_creal_PRECISION(p), p_im = cu_cimag_PRECISION(p), o_re, o_im;
      restrict_op_entry_PRECISION<SSE_LAYOUT>( op, row, j, nev, vector_size, S, &o_re, &o_im );
      // phi * conj(op)
      re += p_re*o_re + p_im*o_im;
      im += p_im*o_re - p_re*o_im;
    }
  }

  red_re[y*X + j] = re;
  red_im[y*X + j] = im;
  __syncthreads();

  for ( int s=Y/2; s>0; s>>=1 ) {
    if ( y < s ) {
      red_re[y*X + j] += red_re[(y+s)*X + j];
      red_im[y*X + j] += red_im[(y+s)*X + j];
    }
    __syncthreads();
  }

  if ( y == 0 && j < nev )
    phi_c[ (size_t)blockIdx.x*nev + j ] = make_cu_cmplx_PRECISION( red_re[j], red_im[j] );
}


// Interpolation phi (+)= P phi_c. One thread per row of the fine vector:
//   phi[row] (+)= sum_j op(row, j) * phi_c[(2*i + k1)*nev + j]      (no complex conjugate)
// with row = ((i*agg_sites + k)*2 + k1)*npev + k2 (aggregate i, site k, spin half k1).
// add = 0: phi = P phi_c (interpolate3_PRECISION), add = 1: phi += P phi_c (interpolate_PRECISION).
// The sum over j runs in increasing j, as in the plain CPU loop.
template<bool SSE_LAYOUT>
__global__ void cuda_interpolate_PRECISION_kernel( cuda_vector_PRECISION phi, const cu_cmplx_PRECISION *phi_c,
                                                   const cu_cmplx_PRECISION *op, int nev, int npev,
                                                   int agg_sites, size_t n_rows, size_t vector_size, int S,
                                                   int add ) {

  const size_t row = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
  if ( row >= n_rows ) return;

  const size_t b  = row / npev;                      // = (i*agg_sites + k)*2 + k1
  const int    k1 = (int)( b % 2 );
  const size_t i  = b / ( 2*(size_t)agg_sites );
  const cu_cmplx_PRECISION *pc = phi_c + ( 2*i + k1 )*nev;

  PRECISION re = 0, im = 0;
  if ( SSE_LAYOUT ) {
    // SSE layout: test vectors in blocks of S, block jb starts at jb*S*vector_size;
    // each row stores [re_0 .. re_{S-1}, im_0 .. im_{S-1}]
    const PRECISION *opf = (const PRECISION*) op;
    for ( int jb=0; jb*S<nev; jb++ ) {
      const PRECISION *o = opf + 2*( (size_t)jb*S*vector_size + (size_t)S*row );
      for ( int jj=0; jj<S && jb*S+jj<nev; jj++ ) {
        PRECISION o_re = o[jj], o_im = o[S+jj];
        cu_cmplx_PRECISION c = pc[jb*S+jj];
        PRECISION c_re = cu_creal_PRECISION(c), c_im = cu_cimag_PRECISION(c);
        // op * phi_c
        re += o_re*c_re - o_im*c_im;
        im += o_re*c_im + o_im*c_re;
      }
    }
  } else {
    // plain layout (interpolation_generic.c): interleaved complex numbers, test vector index fastest
    const cu_cmplx_PRECISION *o = op + row*nev;
    for ( int j=0; j<nev; j++ ) {
      cu_cmplx_PRECISION oo = o[j], c = pc[j];
      PRECISION o_re = cu_creal_PRECISION(oo), o_im = cu_cimag_PRECISION(oo),
                c_re = cu_creal_PRECISION(c),  c_im = cu_cimag_PRECISION(c);
      re += o_re*c_re - o_im*c_im;
      im += o_re*c_im + o_im*c_re;
    }
  }

  if ( add ) {
    cu_cmplx_PRECISION p = phi[row];
    re += cu_creal_PRECISION(p);
    im += cu_cimag_PRECISION(p);
  }
  phi[row] = make_cu_cmplx_PRECISION( re, im );
}


// number of SMs of the current device (queried once)
static int restrict_num_sms_PRECISION( void ) {
  static int num_sms = 0;
  if ( num_sms == 0 ) {
    int dev;
    CUDA_CHECK_INTERP( cudaGetDevice( &dev ) );
    CUDA_CHECK_INTERP( cudaDeviceGetAttribute( &num_sms, cudaDevAttrMultiProcessorCount, dev ) );
  }
  return num_sms;
}


// launches the restriction kernel for the given operator layout
template<bool SSE_LAYOUT>
static void cuda_restrict_PRECISION_launch( cuda_vector_PRECISION phi_c_gpu, const cu_cmplx_PRECISION *phi_gpu,
                                            const cu_cmplx_PRECISION *op_gpu, level_struct *l ) {

  int num_agg   = l->is_PRECISION.num_agg,
      nev       = l->num_eig_vect,
      npev      = l->num_lattice_site_var/2,
      agg_sites = l->num_inner_lattice_sites / num_agg,
      n_rows    = agg_sites*npev;

  // block shape: X = nev padded to a multiple of 32, Y = power of 2 with X*Y <= 1024,
  // at least 4, grown until the grid fills ~2048 threads per SM (or Y reaches the number of rows)
  int X = ((nev + 31)/32)*32;
  if ( X > 1024 ) {
    fprintf( stderr, "cuda_restrict_PRECISION: num_eig_vect=%d too large for one block\n", nev );
    MPI_Abort( MPI_COMM_WORLD, 1 );
  }
  int y_max = 1024 / X;
  long target = 2048L * restrict_num_sms_PRECISION();
  int Y = 1;
  while ( 2*Y <= y_max && ( Y < 4 || ( (long)2*num_agg*X*Y < target && Y < n_rows ) ) ) Y *= 2;

  dim3 block( X, Y );
  int nr_blocks = 2*num_agg;
  size_t smem = 2*(size_t)X*Y*sizeof(PRECISION);

  cuda_restrict_PRECISION_kernel<SSE_LAYOUT><<< nr_blocks, block, smem >>>( phi_c_gpu, phi_gpu, op_gpu, nev, npev,
                                                                            agg_sites, (size_t)l->vector_size,
                                                                            restrict_simd_length_PRECISION() );
  CUDA_CHECK_INTERP( cudaGetLastError() );
}


// launches the interpolation kernel for the given operator layout (one thread per fine row)
template<bool SSE_LAYOUT>
static void cuda_interpolate_PRECISION_launch( cuda_vector_PRECISION phi_gpu, const cu_cmplx_PRECISION *phi_c_gpu,
                                               const cu_cmplx_PRECISION *op_gpu, level_struct *l, int add ) {

  int num_agg   = l->is_PRECISION.num_agg,
      nev       = l->num_eig_vect,
      npev      = l->num_lattice_site_var/2,
      agg_sites = l->num_inner_lattice_sites / num_agg;
  size_t n_rows = (size_t)l->inner_vector_size;

  const int threads = 256;
  unsigned int nr_blocks = (unsigned int)( (n_rows + threads - 1) / threads );

  cuda_interpolate_PRECISION_kernel<SSE_LAYOUT><<< nr_blocks, threads >>>( phi_gpu, phi_c_gpu, op_gpu, nev, npev,
                                                                           agg_sites, n_rows, (size_t)l->vector_size,
                                                                           restrict_simd_length_PRECISION(), add );
  CUDA_CHECK_INTERP( cudaGetLastError() );
}


#if defined( CUDA_RESTRICT_TEST ) || defined( CUDA_INTERP_TEST )

// entry (row, j) of the host operator, read in the native layout of this build
static void host_op_entry_PRECISION( const PRECISION *opf, size_t row, int j, int nev, size_t vector_size, int S,
                                     PRECISION *re, PRECISION *im ) {
  if ( restrict_sse_layout_PRECISION ) {
    size_t base = 2*( (size_t)(j/S)*S*vector_size + (size_t)S*row );
    *re = opf[ base + j%S ];
    *im = opf[ base + S + j%S ];
  } else {
    *re = opf[ 2*(row*nev + j) ];
    *im = opf[ 2*(row*nev + j) + 1 ];
  }
}

// host copy of the operator in the plain layout op_plain[row*nev + j] (2 reals per entry)
static PRECISION *host_op_plain_PRECISION( level_struct *l ) {
  size_t n_rows = (size_t)l->inner_vector_size;
  int nev = l->num_eig_vect, S = restrict_simd_length_PRECISION();
  const PRECISION *opf = (const PRECISION*) l->is_PRECISION.op;
  PRECISION *op_plain = (PRECISION*) malloc( 2*n_rows*nev*sizeof(PRECISION) );
  if ( op_plain == NULL ) {
    fprintf( stderr, "cuda interpolation test: host allocation failed\n" );
    MPI_Abort( MPI_COMM_WORLD, 1 );
  }
  for ( size_t row=0; row<n_rows; row++ )
    for ( int j=0; j<nev; j++ )
      host_op_entry_PRECISION( opf, row, j, nev, (size_t)l->vector_size, S,
                               &op_plain[2*(row*nev+j)], &op_plain[2*(row*nev+j)+1] );
  return op_plain;
}

#endif


#ifdef CUDA_RESTRICT_TEST

// one test per level (indexed by depth)
static int restrict_tested_PRECISION[64] = {0};

// Unit test of the plain-layout kernel (the one used in builds without SSE):
//   1. converts the host operator to the plain layout of interpolation_generic.c
//   2. computes the restriction on the CPU with the loop of interpolation_generic.c
//   3. runs the plain-layout instance of the GPU kernel
//   4. compares both GPU results (plain-layout test and production) with the CPU reference
// Expects phi already uploaded to l->is_PRECISION.eta_gpu and the production result in the transfer buffer.
static void cuda_restrict_PRECISION_plain_layout_test( vector_PRECISION phi, level_struct *l ) {

  int num_agg   = l->is_PRECISION.num_agg,
      nev       = l->num_eig_vect,
      npev      = l->num_lattice_site_var/2,
      agg_sites = l->num_inner_lattice_sites / num_agg,
      S         = restrict_simd_length_PRECISION();
  size_t n_rows   = (size_t)l->inner_vector_size,
         n_coarse = (size_t)num_agg*2*nev,
         n_op     = n_rows*nev;

  const PRECISION *opf  = (const PRECISION*) l->is_PRECISION.op;
  const PRECISION *phif = (const PRECISION*) phi;
  const PRECISION *prod = (const PRECISION*) l->next_level->gs_PRECISION.transfer_buffer;

  PRECISION *op_plain = (PRECISION*) malloc( 2*n_op*sizeof(PRECISION) );
  PRECISION *ref      = (PRECISION*) malloc( 2*n_coarse*sizeof(PRECISION) );
  PRECISION *res_gpu  = (PRECISION*) malloc( 2*n_coarse*sizeof(PRECISION) );
  if ( op_plain == NULL || ref == NULL || res_gpu == NULL ) {
    fprintf( stderr, "restrict test: host allocation failed\n" );
    MPI_Abort( MPI_COMM_WORLD, 1 );
  }

  // 1. operator in the plain layout: op_plain[row*nev + j]
  for ( size_t row=0; row<n_rows; row++ )
    for ( int j=0; j<nev; j++ )
      host_op_entry_PRECISION( opf, row, j, nev, (size_t)l->vector_size, S,
                               &op_plain[2*(row*nev+j)], &op_plain[2*(row*nev+j)+1] );

  // 2. CPU reference: restriction loop of interpolation_generic.c (real arithmetic)
  int sign = 1;
  for ( int i=0; i<num_agg; i++ ) {
    const PRECISION *phi_pt = phif + 2*(size_t)i*2*npev*agg_sites;
    PRECISION *phi_c_pt     = ref  + 2*(size_t)i*2*nev;
    const PRECISION *op_pt  = op_plain + 2*(size_t)i*2*nev*npev*agg_sites;

    for ( int j=0; j<2*nev; j++ ) {
      phi_c_pt[2*j] = 0; phi_c_pt[2*j+1] = 0;
    }

    for ( int k=0; k<agg_sites; k++ ) {
      for ( int k1=0; k1<2; k1++ ) {
        for ( int k2=0; k2<npev; k2++ ) {
          for ( int j=0; j<nev; j++ ) {
            // phi * conj(op)
            phi_c_pt[2*j]   += phi_pt[0]*op_pt[0] + phi_pt[1]*op_pt[1];
            phi_c_pt[2*j+1] += phi_pt[1]*op_pt[0] - phi_pt[0]*op_pt[1];
            op_pt += 2;
          }
          phi_pt += 2;
        }
        phi_c_pt += 2*sign*nev; sign *= -1;
      }
    }
  }

  // 3. GPU: plain-layout instance of the production kernel
  cu_cmplx_PRECISION *op_plain_gpu = NULL, *phi_c_test_gpu = NULL;
  CUDA_CHECK_INTERP( cudaMalloc( (void**)&op_plain_gpu,   n_op*sizeof(cu_cmplx_PRECISION) ) );
  CUDA_CHECK_INTERP( cudaMalloc( (void**)&phi_c_test_gpu, n_coarse*sizeof(cu_cmplx_PRECISION) ) );
  CUDA_CHECK_INTERP( cudaMemcpy( op_plain_gpu, op_plain, n_op*sizeof(cu_cmplx_PRECISION), cudaMemcpyHostToDevice ) );
  cuda_restrict_PRECISION_launch<false>( phi_c_test_gpu, l->is_PRECISION.eta_gpu, op_plain_gpu, l );
  CUDA_CHECK_INTERP( cudaMemcpy( res_gpu, phi_c_test_gpu, n_coarse*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost ) );
  CUDA_CHECK_INTERP( cudaFree( op_plain_gpu ) );
  CUDA_CHECK_INTERP( cudaFree( phi_c_test_gpu ) );

  // 4. compare (max abs difference relative to the max abs entry of the reference)
  double max_ref = 0, diff_plain = 0, diff_prod = 0;
  for ( size_t m=0; m<2*n_coarse; m++ ) {
    max_ref    = fmax( max_ref,    fabs( (double)ref[m] ) );
    diff_plain = fmax( diff_plain, fabs( (double)res_gpu[m] - (double)ref[m] ) );
    diff_prod  = fmax( diff_prod,  fabs( (double)prod[m]    - (double)ref[m] ) );
  }
  if ( max_ref > 0 ) { diff_plain /= max_ref; diff_prod /= max_ref; }

  double tol = ( sizeof(PRECISION) == sizeof(float) ) ? 1e-4 : 1e-10;
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  printf( "[rank %d] restrict test (depth %d, %s, build layout: %s):\n"
          "[rank %d]   plain-layout GPU kernel vs CPU reference: max rel. diff = %.3e  %s\n"
          "[rank %d]   production GPU kernel   vs CPU reference: max rel. diff = %.3e  %s\n",
          rank, l->depth, ( sizeof(PRECISION) == sizeof(float) ) ? "float" : "double",
          restrict_sse_layout_PRECISION ? "SSE" : "plain",
          rank, diff_plain, ( diff_plain < tol ) ? "PASS" : "FAIL",
          rank, diff_prod,  ( diff_prod  < tol ) ? "PASS" : "FAIL" );
  fflush( stdout );

  free( op_plain );
  free( ref );
  free( res_gpu );
}

#endif // CUDA_RESTRICT_TEST


#ifdef CUDA_INTERP_TEST

// one test per level and variant (indexed by depth, add)
static int interp_tested_PRECISION[64][2] = {{0}};

// Unit test of the interpolation (once per depth and variant):
//   1. CPU reference with the loop of interpolation_generic.c (plain layout, double accumulation)
//   2. plain-layout instance of the GPU kernel
//   3. compares both GPU results (plain-layout test and production, phi_out) with the reference
// phi_in: copy of phi before the call (used when add = 1); phi_c: distributed coarse vector
// (transfer buffer), already on the GPU in l->is_PRECISION.phi_c_gpu.
static void cuda_interpolate_PRECISION_test( const cu_cmplx_PRECISION *phi_in, vector_PRECISION phi_out,
                                             int add, level_struct *l ) {

  int num_agg   = l->is_PRECISION.num_agg,
      nev       = l->num_eig_vect,
      npev      = l->num_lattice_site_var/2,
      agg_sites = l->num_inner_lattice_sites / num_agg;
  size_t n_rows = (size_t)l->inner_vector_size;

  PRECISION *op_plain = host_op_plain_PRECISION( l );
  const PRECISION *pc   = (const PRECISION*) l->next_level->gs_PRECISION.transfer_buffer;
  const PRECISION *pin  = (const PRECISION*) phi_in;
  const PRECISION *pout = (const PRECISION*) phi_out;
  PRECISION *ref     = (PRECISION*) malloc( 2*n_rows*sizeof(PRECISION) );
  PRECISION *res_gpu = (PRECISION*) malloc( 2*n_rows*sizeof(PRECISION) );
  if ( ref == NULL || res_gpu == NULL ) {
    fprintf( stderr, "interpolate test: host allocation failed\n" );
    MPI_Abort( MPI_COMM_WORLD, 1 );
  }

  // 1. CPU reference: loop of interpolation_generic.c
  for ( int i=0; i<num_agg; i++ )
    for ( int k=0; k<agg_sites; k++ )
      for ( int k1=0; k1<2; k1++ )
        for ( int k2=0; k2<npev; k2++ ) {
          size_t row = ( ( (size_t)i*agg_sites + k )*2 + k1 )*npev + k2;
          const PRECISION *c = pc + 2*( (size_t)2*i + k1 )*nev;
          double re = add ? pin[2*row] : 0, im = add ? pin[2*row+1] : 0;
          for ( int j=0; j<nev; j++ ) {
            double o_re = op_plain[2*(row*nev+j)], o_im = op_plain[2*(row*nev+j)+1];
            re += o_re*c[2*j] - o_im*c[2*j+1];
            im += o_re*c[2*j+1] + o_im*c[2*j];
          }
          ref[2*row] = (PRECISION) re; ref[2*row+1] = (PRECISION) im;
        }

  // 2. GPU: plain-layout instance of the production kernel
  cu_cmplx_PRECISION *op_plain_gpu = NULL, *phi_test_gpu = NULL;
  CUDA_CHECK_INTERP( cudaMalloc( (void**)&op_plain_gpu, n_rows*nev*sizeof(cu_cmplx_PRECISION) ) );
  CUDA_CHECK_INTERP( cudaMalloc( (void**)&phi_test_gpu, n_rows*sizeof(cu_cmplx_PRECISION) ) );
  CUDA_CHECK_INTERP( cudaMemcpy( op_plain_gpu, op_plain, n_rows*nev*sizeof(cu_cmplx_PRECISION), cudaMemcpyHostToDevice ) );
  CUDA_CHECK_INTERP( cudaMemcpy( phi_test_gpu, phi_in, n_rows*sizeof(cu_cmplx_PRECISION), cudaMemcpyHostToDevice ) );
  cuda_interpolate_PRECISION_launch<false>( phi_test_gpu, l->is_PRECISION.phi_c_gpu, op_plain_gpu, l, add );
  CUDA_CHECK_INTERP( cudaMemcpy( res_gpu, phi_test_gpu, n_rows*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost ) );
  CUDA_CHECK_INTERP( cudaFree( op_plain_gpu ) );
  CUDA_CHECK_INTERP( cudaFree( phi_test_gpu ) );

  // 3. compare (max abs difference relative to the max abs entry of the reference)
  double max_ref = 0, diff_plain = 0, diff_prod = 0;
  for ( size_t m=0; m<2*n_rows; m++ ) {
    max_ref    = fmax( max_ref,    fabs( (double)ref[m] ) );
    diff_plain = fmax( diff_plain, fabs( (double)res_gpu[m] - (double)ref[m] ) );
    diff_prod  = fmax( diff_prod,  fabs( (double)pout[m]    - (double)ref[m] ) );
  }
  if ( max_ref > 0 ) { diff_plain /= max_ref; diff_prod /= max_ref; }

  double tol = ( sizeof(PRECISION) == sizeof(float) ) ? 1e-4 : 1e-10;
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  printf( "[rank %d] interpolate test (depth %d, %s, %s, build layout: %s):\n"
          "[rank %d]   plain-layout GPU kernel vs CPU reference: max rel. diff = %.3e  %s\n"
          "[rank %d]   production GPU kernel   vs CPU reference: max rel. diff = %.3e  %s\n",
          rank, l->depth, add ? "phi += P phi_c" : "phi = P phi_c",
          ( sizeof(PRECISION) == sizeof(float) ) ? "float" : "double",
          restrict_sse_layout_PRECISION ? "SSE" : "plain",
          rank, diff_plain, ( diff_plain < tol ) ? "PASS" : "FAIL",
          rank, diff_prod,  ( diff_prod  < tol ) ? "PASS" : "FAIL" );
  fflush( stdout );

  free( op_plain );
  free( ref );
  free( res_gpu );
}

#endif // CUDA_INTERP_TEST


// copies l->is_PRECISION.op to the GPU (allocates device buffers at first call)
// to be called by one thread only, once the operator is final
extern "C" void cuda_interpolation_PRECISION_op_to_gpu( level_struct *l ) {

  size_t op_size  = interpolation_op_size_PRECISION( l ),
         n_fine   = (size_t)l->inner_vector_size,
         n_coarse = (size_t)l->is_PRECISION.num_agg*2*l->num_eig_vect;

  if ( l->is_PRECISION.op_gpu == NULL ) {
    CUDA_CHECK_INTERP( cudaMalloc( (void**)&(l->is_PRECISION.op_gpu),    op_size*sizeof(cu_cmplx_PRECISION) ) );
    CUDA_CHECK_INTERP( cudaMalloc( (void**)&(l->is_PRECISION.phi_gpu),   n_fine*sizeof(cu_cmplx_PRECISION) ) );
    CUDA_CHECK_INTERP( cudaMalloc( (void**)&(l->is_PRECISION.phi_c_gpu), n_coarse*sizeof(cu_cmplx_PRECISION) ) );
    CUDA_CHECK_INTERP( cudaMalloc( (void**)&(l->is_PRECISION.eta_gpu),   n_fine*sizeof(cu_cmplx_PRECISION) ) );
  }

  CUDA_CHECK_INTERP( cudaMemcpy( l->is_PRECISION.op_gpu, l->is_PRECISION.op,
                                 op_size*sizeof(cu_cmplx_PRECISION), cudaMemcpyHostToDevice ) );
}


extern "C" void cuda_interpolation_PRECISION_free( level_struct *l ) {

  if ( l->is_PRECISION.op_gpu != NULL ) {
    CUDA_CHECK_INTERP( cudaFree( l->is_PRECISION.op_gpu ) );
    CUDA_CHECK_INTERP( cudaFree( l->is_PRECISION.phi_gpu ) );
    CUDA_CHECK_INTERP( cudaFree( l->is_PRECISION.phi_c_gpu ) );
    CUDA_CHECK_INTERP( cudaFree( l->is_PRECISION.eta_gpu ) );
    if ( l->is_PRECISION.oe_b_gpu != NULL ) {
      CUDA_CHECK_INTERP( cudaFree( l->is_PRECISION.oe_b_gpu ) );
      CUDA_CHECK_INTERP( cudaFree( l->is_PRECISION.oe_x_gpu ) );
      CUDA_CHECK_INTERP( cudaFree( l->is_PRECISION.oe_chunk_gpu ) );
      CUDA_CHECK_INTERP( cudaFree( l->is_PRECISION.oe_perm_gpu ) );
      if ( l->is_PRECISION.oe_V_gpu != NULL ) {
        CUDA_CHECK_INTERP( cudaFree( l->is_PRECISION.oe_V_gpu ) );
        CUDA_CHECK_INTERP( cudaFree( l->is_PRECISION.oe_h_gpu ) );
        l->is_PRECISION.oe_V_gpu = NULL; l->is_PRECISION.oe_h_gpu = NULL; l->is_PRECISION.oe_V_cols = 0;
      }
      l->is_PRECISION.oe_b_gpu = NULL; l->is_PRECISION.oe_x_gpu = NULL;
      l->is_PRECISION.oe_chunk_gpu = NULL; l->is_PRECISION.oe_perm_gpu = NULL;
    }
    l->is_PRECISION.op_gpu    = NULL;
    l->is_PRECISION.phi_gpu   = NULL;
    l->is_PRECISION.phi_c_gpu = NULL;
    l->is_PRECISION.eta_gpu   = NULL;
  }
  l->is_PRECISION.phi_gpu_host = NULL;
  l->is_PRECISION.eta_gpu_host = NULL;
  l->is_PRECISION.defer_phi_dtoh = 0;
  l->is_PRECISION.phi_stale_host = NULL;
  l->is_PRECISION.eta_resident_host = NULL;
  l->is_PRECISION.eta_stale_host = NULL;
  l->is_PRECISION.keep_phi_result = 0;
  l->is_PRECISION.phi_result_host = NULL;
}


// GPU version of restrict_PRECISION: same interface, phi and phi_c are host vectors
extern "C" void cuda_restrict_PRECISION( vector_PRECISION phi_c, vector_PRECISION phi, level_struct *l, struct Thread *threading ) {

  SYNC_CORES(threading)
  SYNC_HYPERTHREADS(threading)

  PROF_PRECISION_START( _PR, threading );

  START_LOCKED_MASTER(threading)

  size_t n_fine   = (size_t)l->inner_vector_size,
         n_coarse = (size_t)l->is_PRECISION.num_agg*2*l->num_eig_vect;

  if ( l->is_PRECISION.op_gpu == NULL ) {
    fprintf( stderr, "cuda_restrict_PRECISION: interpolation operator not on GPU (depth=%d)\n", l->depth );
    MPI_Abort( MPI_COMM_WORLD, 1 );
  }

  if ( l->is_PRECISION.eta_resident_host == phi ) {
    // eta was written into eta_gpu by the outer FGMRES on the GPU: no copy, the host copy is stale
    l->is_PRECISION.eta_stale_host = phi;
  } else {
    CUDA_CHECK_INTERP( cudaMemcpy( l->is_PRECISION.eta_gpu, phi, n_fine*sizeof(cu_cmplx_PRECISION),
                                   cudaMemcpyHostToDevice ) );
    l->is_PRECISION.eta_stale_host = NULL;
  }
  l->is_PRECISION.eta_resident_host = NULL;
  l->is_PRECISION.eta_gpu_host = phi;   // kept on the GPU for the smoother residual

  cuda_restrict_PRECISION_launch<restrict_sse_layout_PRECISION>( l->is_PRECISION.phi_c_gpu, l->is_PRECISION.eta_gpu,
                                                                 l->is_PRECISION.op_gpu, l );

  // synchronous copy on the default stream: waits for the kernel to finish
  CUDA_CHECK_INTERP( cudaMemcpy( l->next_level->gs_PRECISION.transfer_buffer, l->is_PRECISION.phi_c_gpu,
                                 n_coarse*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost ) );

#ifdef CUDA_RESTRICT_TEST
  if ( l->depth < 64 && !restrict_tested_PRECISION[l->depth] ) {
    restrict_tested_PRECISION[l->depth] = 1;
    cuda_restrict_PRECISION_plain_layout_test( phi, l );
  }
#endif

  // distribution to the coarse-level processes stays on the CPU (MPI)
  vector_PRECISION_gather( phi_c, l->next_level->gs_PRECISION.transfer_buffer, l->next_level );

  END_LOCKED_MASTER(threading)

  PROF_PRECISION_STOP( _PR, 1, threading );
}


// GPU version of interpolate_PRECISION (add = 1, phi += P phi_c) and interpolate3_PRECISION
// (add = 0, phi = P phi_c): same interface, phi and phi_c are host vectors
extern "C" void cuda_interpolate_PRECISION( vector_PRECISION phi, vector_PRECISION phi_c, level_struct *l,
                                            struct Thread *threading, int add ) {

  // phi may have been written by all threads (add = 1)
  SYNC_CORES(threading)
  SYNC_HYPERTHREADS(threading)

  PROF_PRECISION_START( _PR, threading );

  START_LOCKED_MASTER(threading)

  size_t n_fine   = (size_t)l->inner_vector_size,
         n_coarse = (size_t)l->is_PRECISION.num_agg*2*l->num_eig_vect;

  if ( l->is_PRECISION.op_gpu == NULL ) {
    fprintf( stderr, "cuda_interpolate_PRECISION: interpolation operator not on GPU (depth=%d)\n", l->depth );
    MPI_Abort( MPI_COMM_WORLD, 1 );
  }

  // distribution from the coarse-level processes stays on the CPU (MPI)
  vector_PRECISION_distribute( l->next_level->gs_PRECISION.transfer_buffer, phi_c, l->next_level );

#ifdef CUDA_INTERP_TEST
  cu_cmplx_PRECISION *phi_in = NULL;
  int do_test = ( l->depth < 64 && !interp_tested_PRECISION[l->depth][add?1:0] );
  if ( do_test ) {
    interp_tested_PRECISION[l->depth][add?1:0] = 1;
    phi_in = (cu_cmplx_PRECISION*) malloc( n_fine*sizeof(cu_cmplx_PRECISION) );
    if ( phi_in == NULL ) {
      fprintf( stderr, "interpolate test: host allocation failed\n" );
      MPI_Abort( MPI_COMM_WORLD, 1 );
    }
    memcpy( phi_in, phi, n_fine*sizeof(cu_cmplx_PRECISION) );
  }
#endif

  CUDA_CHECK_INTERP( cudaMemcpy( l->is_PRECISION.phi_c_gpu, l->next_level->gs_PRECISION.transfer_buffer,
                                 n_coarse*sizeof(cu_cmplx_PRECISION), cudaMemcpyHostToDevice ) );
  if ( add )
    CUDA_CHECK_INTERP( cudaMemcpy( l->is_PRECISION.phi_gpu, phi, n_fine*sizeof(cu_cmplx_PRECISION),
                                   cudaMemcpyHostToDevice ) );

  cuda_interpolate_PRECISION_launch<restrict_sse_layout_PRECISION>( l->is_PRECISION.phi_gpu, l->is_PRECISION.phi_c_gpu,
                                                                    l->is_PRECISION.op_gpu, l, add );

#ifdef CUDA_INTERP_TEST
  l->is_PRECISION.defer_phi_dtoh = 0;   // the test compares on the host
#endif
  if ( l->is_PRECISION.defer_phi_dtoh ) {
    // the depth-0 smoother continues from phi_gpu and copies phi to the host at its end
    CUDA_CHECK_INTERP( cudaDeviceSynchronize() );
    l->is_PRECISION.phi_stale_host = phi;
  } else {
    // synchronous copy on the default stream: waits for the kernel to finish
    CUDA_CHECK_INTERP( cudaMemcpy( phi, l->is_PRECISION.phi_gpu, n_fine*sizeof(cu_cmplx_PRECISION),
                                   cudaMemcpyDeviceToHost ) );
    l->is_PRECISION.phi_stale_host = NULL;
  }
  l->is_PRECISION.defer_phi_dtoh = 0;
  l->is_PRECISION.phi_gpu_host = phi;   // kept on the GPU for the smoother residual

#ifdef CUDA_INTERP_TEST
  if ( do_test ) {
    cuda_interpolate_PRECISION_test( phi_in, phi, add, l );
    free( phi_in );
  }
#endif

  END_LOCKED_MASTER(threading)

  PROF_PRECISION_STOP( _PR, 1, threading );
}


__global__ void cuda_fine_residual_PRECISION_kernel( cu_cmplx_PRECISION *r, const cu_cmplx_PRECISION *eta,
                                                     const cu_cmplx_PRECISION *dphi, size_t n ) {
  size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
  if ( i < n )
    r[i] = make_cu_cmplx_PRECISION( eta[i].x - dphi[i].x, eta[i].y - dphi[i].y );
}


// Depth-0 smoother residual r = eta - D phi on the GPU, using the copies of phi (left on the GPU
// by the interpolation) and of eta (left by the restriction) if they still belong to these host
// vectors. Returns 1 if the residual was computed here (r is then on the host, or in op->x_gpu
// if keep_on_gpu), 0 if the caller
// has to compute it. Called by all threads; the tags are consumed in both cases.
extern "C" int cuda_fine_residual_PRECISION( vector_PRECISION r, vector_PRECISION phi, vector_PRECISION eta,
                                             operator_PRECISION_struct *op, level_struct *l,
                                             struct Thread *threading, int keep_on_gpu ) {

  interpolation_PRECISION_struct *is = &(l->is_PRECISION);
  int use = ( l->depth == 0 && is->op_gpu != NULL && op->x_componentwise_gpu != NULL &&
              is->phi_gpu_host == phi && is->eta_gpu_host == eta );

  // every thread has read the tags before the master resets them
  SYNC_CORES(threading)
  SYNC_HYPERTHREADS(threading)

  START_LOCKED_MASTER(threading)
  if ( is->phi_stale_host != NULL ) {
    // phi was left only on the GPU by the interpolation: the host copy is needed by the CPU
    // path (fallback) and by the smoothers that do not continue on the GPU
    if ( !( use && keep_on_gpu ) )
      CUDA_CHECK_INTERP( cudaMemcpy( is->phi_stale_host, is->phi_gpu,
                                     (size_t)l->inner_vector_size*sizeof(cu_cmplx_PRECISION),
                                     cudaMemcpyDeviceToHost ) );
    is->phi_stale_host = NULL;
  }
  if ( is->eta_stale_host != NULL ) {
    // eta was given to the restriction only on the GPU (outer FGMRES on the GPU)
    if ( !( use && keep_on_gpu ) )
      CUDA_CHECK_INTERP( cudaMemcpy( is->eta_stale_host, is->eta_gpu,
                                     (size_t)l->inner_vector_size*sizeof(cu_cmplx_PRECISION),
                                     cudaMemcpyDeviceToHost ) );
    is->eta_stale_host = NULL;
  }
  if ( use ) {
    const size_t n_sites = (size_t)l->num_inner_lattice_sites,
                 nsv     = (size_t)l->num_lattice_site_var,
                 n_fine  = (size_t)l->inner_vector_size;
    const unsigned int block = 128;

    // phi (ordering of the Schwarz operator, site by site) -> component-wise, D phi, back
    reorderArrayByComponent<<< minGridSizeForN( n_sites, block ), block >>>( op->x_componentwise_gpu, is->phi_gpu,
                                                                             nsv, n_sites );
    CUDA_CHECK_INTERP( cudaDeviceSynchronize() );
    cuda_d_plus_clover_PRECISION( op->w_componentwise_gpu, op->x_componentwise_gpu, op, l, threading );
    reorderArrayByChunks<<< minGridSizeForN( n_sites, block ), block >>>( op->w_gpu, op->w_componentwise_gpu,
                                                                          nsv, n_sites );
    // r = eta - D phi, in op->x_gpu (free at this point)
    cuda_fine_residual_PRECISION_kernel<<< minGridSizeForN( n_fine, 256 ), 256 >>>( op->x_gpu, is->eta_gpu,
                                                                                   op->w_gpu, n_fine );
    CUDA_CHECK_INTERP( cudaGetLastError() );
    if ( keep_on_gpu ) {
      // r stays in op->x_gpu for the odd-even part of the smoother (cuda_smoother_oe_prepare)
      CUDA_CHECK_INTERP( cudaDeviceSynchronize() );
    } else {
      // synchronous copy: waits for the kernels
      CUDA_CHECK_INTERP( cudaMemcpy( r, op->x_gpu, n_fine*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost ) );
    }
  }
  is->phi_gpu_host = NULL;
  is->eta_gpu_host = NULL;
  END_LOCKED_MASTER(threading)
  SYNC_HYPERTHREADS(threading)

  return use;
}


// Schwarz layout (site by site) -> odd-even layout, component-wise within the even and the odd half
__global__ void cuda_block_to_oe_cw_PRECISION_kernel( cu_cmplx_PRECISION *out_cw, const cu_cmplx_PRECISION *in,
                                                      const int *perm, int n_sites, int n_even, int nsv ) {
  int m = blockIdx.x*blockDim.x + threadIdx.x;
  if ( m >= n_sites ) return;
  int k = perm[m];
  size_t stride = ( k < n_even ) ? (size_t)n_even : (size_t)(n_sites - n_even);
  cu_cmplx_PRECISION *dst = ( k < n_even ) ? out_cw + k : out_cw + (size_t)n_even*nsv + (k - n_even);
  const cu_cmplx_PRECISION *src = in + (size_t)m*nsv;
  for ( int c=0; c<nsv; c++ )
    dst[c*stride] = src[c];
}

// phi (Schwarz layout) += x (odd-even layout, component-wise)
__global__ void cuda_oe_cw_add_to_block_PRECISION_kernel( cu_cmplx_PRECISION *phi, const cu_cmplx_PRECISION *x_cw,
                                                          const int *perm, int n_sites, int n_even, int nsv ) {
  int m = blockIdx.x*blockDim.x + threadIdx.x;
  if ( m >= n_sites ) return;
  int k = perm[m];
  size_t stride = ( k < n_even ) ? (size_t)n_even : (size_t)(n_sites - n_even);
  const cu_cmplx_PRECISION *src = ( k < n_even ) ? x_cw + k : x_cw + (size_t)n_even*nsv + (k - n_even);
  cu_cmplx_PRECISION *dst = phi + (size_t)m*nsv;
  for ( int c=0; c<nsv; c++ ) {
    cu_cmplx_PRECISION a = dst[c], b = src[c*stride];
    dst[c] = make_cu_cmplx_PRECISION( a.x + b.x, a.y + b.y );
  }
}


// depth-0 odd-even buffers and the Schwarz -> odd-even site permutation, allocated on first use
static void smoother_oe_buffers_PRECISION( level_struct *l ) {

  interpolation_PRECISION_struct *is = &(l->is_PRECISION);
  if ( is->oe_b_gpu != NULL ) return;

  const int n = l->num_inner_lattice_sites;
  const size_t n_fine = (size_t)l->inner_vector_size;
  const int *tt_oe = l->oe_op_PRECISION.translation_table, *tt_b = l->s_PRECISION.op.translation_table;
  int *perm = (int*) malloc( n*sizeof(int) );
  if ( perm == NULL || tt_oe == NULL || tt_b == NULL ) {
    fprintf( stderr, "smoother_oe_buffers_PRECISION: translation tables or host memory not available\n" );
    MPI_Abort( MPI_COMM_WORLD, 1 );
  }
  // same mapping as block_to_oddeven_PRECISION: out[tt_oe[i]] = in[tt_b[i]]
  for ( int i=0; i<n; i++ )
    perm[ tt_b[i] ] = tt_oe[i];

  CUDA_CHECK_INTERP( cudaMalloc( (void**)&(is->oe_perm_gpu),  n*sizeof(int) ) );
  CUDA_CHECK_INTERP( cudaMalloc( (void**)&(is->oe_b_gpu),     n_fine*sizeof(cu_cmplx_PRECISION) ) );
  CUDA_CHECK_INTERP( cudaMalloc( (void**)&(is->oe_x_gpu),     n_fine*sizeof(cu_cmplx_PRECISION) ) );
  CUDA_CHECK_INTERP( cudaMalloc( (void**)&(is->oe_chunk_gpu), n_fine*sizeof(cu_cmplx_PRECISION) ) );
  CUDA_CHECK_INTERP( cudaMemcpy( is->oe_perm_gpu, perm, n*sizeof(int), cudaMemcpyHostToDevice ) );
  free( perm );
}


// odd-even part of the depth-0 smoother, before the GMRES: r (in the GPU buffer x_gpu of the
// Schwarz operator, left there by cuda_fine_residual_PRECISION with keep_on_gpu) -> odd-even
// layout in oe_b_gpu, and preparation of the Schur source b_e = r_e - D_eo D_oo^-1 r_o (as
// cuda_solve_oddeven_PRECISION). r_o stays in the odd half of oe_b_gpu for the reconstruction.
static void smoother_oe_prepare_core_PRECISION( level_struct *l ) {

  interpolation_PRECISION_struct *is = &(l->is_PRECISION);
  operator_PRECISION_struct *op = &(l->oe_op_PRECISION), *s_op = l->p_PRECISION.op;
  cudaStream_t stream = cudaStreamPerThread;
  cudaStream_t* const streams = &stream;
  const int n = l->num_inner_lattice_sites, nsv = l->num_lattice_site_var,
            n_even = op->num_even_sites, n_odd = op->num_odd_sites;
  const int start_odd = n_even*nsv, len_odd = n_odd*nsv;
  const unsigned int css = clover_site_size( nsv, l->depth );
  cuda_vector_PRECISION b_cw = is->oe_b_gpu, tmp = op->buffer_gpu[0];

  cuda_block_to_oe_cw_PRECISION_kernel<<< minGridSizeForN( n, 128 ), 128 >>>( b_cw, s_op->x_gpu, is->oe_perm_gpu,
                                                                              n, n_even, nsv );
  CUDA_CHECK_INTERP( cudaDeviceSynchronize() );

  // odd to even: b_e <- b_e + D_eo ( - D_oo^-1 b_o )
  cuda_diag_oo_inv_componentwise_PRECISION( tmp+start_odd, b_cw+start_odd, op->clover_componentwise_gpu+css*n_even,
                                            n_odd, l );
  cuda_vector_PRECISION_scale( tmp, tmp, make_cu_cmplx_PRECISION(-1.0,0.0), start_odd, len_odd, l, _CUDA_SYNC, 0, streams );
  cuda_hopping_term_PRECISION( b_cw, tmp, op, _EVEN_SITES, l );
  CUDA_CHECK_INTERP( cudaDeviceSynchronize() );
}


// odd-even part of the depth-0 smoother, after the GMRES: with x_e in the even half of oe_x_gpu,
// reconstruction of the odd sites x_o = D_oo^-1 ( r_o - D_oe x_e ) (as cuda_solve_oddeven_PRECISION),
// phi += x in the Schwarz layout on the GPU (phi is still there from the interpolation) and one
// copy of phi to the host.
static void smoother_oe_finish_core_PRECISION( vector_PRECISION phi, level_struct *l ) {

  interpolation_PRECISION_struct *is = &(l->is_PRECISION);
  operator_PRECISION_struct *op = &(l->oe_op_PRECISION);
  cudaStream_t stream = cudaStreamPerThread;
  cudaStream_t* const streams = &stream;
  const int n = l->num_inner_lattice_sites, nsv = l->num_lattice_site_var,
            n_even = op->num_even_sites, n_odd = op->num_odd_sites;
  const int start_odd = n_even*nsv, len_odd = n_odd*nsv;
  const unsigned int css = clover_site_size( nsv, l->depth );
  cuda_vector_PRECISION b_cw = is->oe_b_gpu, x_cw = is->oe_x_gpu, tmp = op->buffer_gpu[0];

  // x_o = D_oo^-1 b_o - D_oo^-1 D_oe x_e   (b_o = r_o, untouched by the preparation)
  cuda_diag_oo_inv_componentwise_PRECISION( x_cw+start_odd, b_cw+start_odd, op->clover_componentwise_gpu+css*n_even,
                                            n_odd, l );
  cuda_vector_PRECISION_define( tmp, make_cu_cmplx_PRECISION(0,0), start_odd, len_odd, l, _CUDA_SYNC, 0, streams );
  cuda_hopping_term_PRECISION( tmp, x_cw, op, _ODD_SITES, l );
  cuda_diag_oo_inv_componentwise_PRECISION( b_cw+start_odd, tmp+start_odd, op->clover_componentwise_gpu+css*n_even,
                                            n_odd, l );
  cuda_vector_PRECISION_minus( x_cw, x_cw, b_cw, start_odd, len_odd, l, _CUDA_SYNC, 0, streams );
  CUDA_CHECK_INTERP( cudaDeviceSynchronize() );

  // phi += x (correction), on the GPU, then one copy of phi to the host
  cuda_oe_cw_add_to_block_PRECISION_kernel<<< minGridSizeForN( n, 128 ), 128 >>>( is->phi_gpu, x_cw, is->oe_perm_gpu,
                                                                                  n, n_even, nsv );
  CUDA_CHECK_INTERP( cudaGetLastError() );
  if ( is->keep_phi_result ) {
    // the outer FGMRES on the GPU reads phi from phi_gpu
    CUDA_CHECK_INTERP( cudaDeviceSynchronize() );
    is->phi_result_host = phi;
  } else {
    CUDA_CHECK_INTERP( cudaMemcpy( phi, is->phi_gpu, (size_t)l->inner_vector_size*sizeof(cu_cmplx_PRECISION),
                                   cudaMemcpyDeviceToHost ) );
  }
}


// Step-B entry points (GMRES on the CPU): kept to compare with the GPU GMRES, see
// SMOOTHER_GMRES_ON_CPU in vcycle_generic.c. b_e goes to the host before the GMRES, x_e comes back.
extern "C" void cuda_smoother_oe_prepare_PRECISION( vector_PRECISION b, level_struct *l, struct Thread *threading ) {

  SYNC_CORES(threading)
  SYNC_HYPERTHREADS(threading)
  START_LOCKED_MASTER(threading)
  smoother_oe_buffers_PRECISION( l );
  smoother_oe_prepare_core_PRECISION( l );
  const int n_even = l->oe_op_PRECISION.num_even_sites, nsv = l->num_lattice_site_var;
  reorderArrayByChunks<<< minGridSizeForN( n_even, 128 ), 128 >>>( l->is_PRECISION.oe_chunk_gpu, l->is_PRECISION.oe_b_gpu,
                                                                    (size_t)nsv, (size_t)n_even );
  CUDA_CHECK_INTERP( cudaMemcpy( b, l->is_PRECISION.oe_chunk_gpu, (size_t)n_even*nsv*sizeof(cu_cmplx_PRECISION),
                                 cudaMemcpyDeviceToHost ) );
  END_LOCKED_MASTER(threading)
  SYNC_HYPERTHREADS(threading)
}

extern "C" void cuda_smoother_oe_finish_PRECISION( vector_PRECISION phi, vector_PRECISION x, level_struct *l,
                                                   struct Thread *threading ) {

  SYNC_CORES(threading)
  SYNC_HYPERTHREADS(threading)
  START_LOCKED_MASTER(threading)
  const int n_even = l->oe_op_PRECISION.num_even_sites, nsv = l->num_lattice_site_var;
  CUDA_CHECK_INTERP( cudaMemcpy( l->is_PRECISION.oe_chunk_gpu, x, (size_t)n_even*nsv*sizeof(cu_cmplx_PRECISION),
                                 cudaMemcpyHostToDevice ) );
  reorderArrayByComponent<<< minGridSizeForN( n_even, 128 ), 128 >>>( l->is_PRECISION.oe_x_gpu, l->is_PRECISION.oe_chunk_gpu,
                                                                      (size_t)nsv, (size_t)n_even );
  CUDA_CHECK_INTERP( cudaDeviceSynchronize() );
  smoother_oe_finish_core_PRECISION( phi, l );
  END_LOCKED_MASTER(threading)
  SYNC_HYPERTHREADS(threading)
}


#define CUBLAS_CHECK_INTERP(call)                                                 \
  do {                                                                            \
    cublasStatus_t st_ = (call);                                                  \
    if ( st_ != CUBLAS_STATUS_SUCCESS ) {                                         \
      fprintf( stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__, __LINE__ ); \
      MPI_Abort( MPI_COMM_WORLD, 1 );                                             \
    }                                                                             \
  } while (0)

// cuBLAS calls for both precisions (overloads, so that this generic file works for float and double)
static inline cublasStatus_t blas_gemv( cublasHandle_t h, cublasOperation_t t, int m, int n, const cuFloatComplex *alpha,
                                        const cuFloatComplex *A, int lda, const cuFloatComplex *x,
                                        const cuFloatComplex *beta, cuFloatComplex *y ) {
  return cublasCgemv( h, t, m, n, alpha, A, lda, x, 1, beta, y, 1 ); }
static inline cublasStatus_t blas_gemv( cublasHandle_t h, cublasOperation_t t, int m, int n, const cuDoubleComplex *alpha,
                                        const cuDoubleComplex *A, int lda, const cuDoubleComplex *x,
                                        const cuDoubleComplex *beta, cuDoubleComplex *y ) {
  return cublasZgemv( h, t, m, n, alpha, A, lda, x, 1, beta, y, 1 ); }
static inline cublasStatus_t blas_nrm2( cublasHandle_t h, int n, const cuFloatComplex *x, float *r ) {
  return cublasScnrm2( h, n, x, 1, r ); }
static inline cublasStatus_t blas_nrm2( cublasHandle_t h, int n, const cuDoubleComplex *x, double *r ) {
  return cublasDznrm2( h, n, x, 1, r ); }
static inline cublasStatus_t blas_rscal( cublasHandle_t h, int n, float a, cuFloatComplex *x ) {
  return cublasCsscal( h, n, &a, x, 1 ); }
static inline cublasStatus_t blas_rscal( cublasHandle_t h, int n, double a, cuDoubleComplex *x ) {
  return cublasZdscal( h, n, &a, x, 1 ); }
static inline cublasStatus_t blas_copy( cublasHandle_t h, int n, const cuFloatComplex *x, cuFloatComplex *y ) {
  return cublasCcopy( h, n, x, 1, y, 1 ); }
static inline cublasStatus_t blas_copy( cublasHandle_t h, int n, const cuDoubleComplex *x, cuDoubleComplex *y ) {
  return cublasZcopy( h, n, x, 1, y, 1 ); }
static inline cublasStatus_t blas_axpy( cublasHandle_t h, int n, const cuFloatComplex *a, const cuFloatComplex *x,
                                        cuFloatComplex *y ) {
  return cublasCaxpy( h, n, a, x, 1, y, 1 ); }
static inline cublasStatus_t blas_axpy( cublasHandle_t h, int n, const cuDoubleComplex *a, const cuDoubleComplex *x,
                                        cuDoubleComplex *y ) {
  return cublasZaxpy( h, n, a, x, 1, y, 1 ); }

static cublasHandle_t smoother_cublas_PRECISION = NULL;


// GMRES on the Schur complement of the depth-0 smoother, entirely on the GPU. b and x are the even
// halves of oe_b_gpu and oe_x_gpu (component-wise layout, as used by the fork's Schur complement).
// Same algorithm as fgmres_PRECISION for this smoother: no preconditioner, classical Gram-Schmidt,
// restart_length steps per restart, num_restart restarts, stop if the estimated relative residual
// drops below p->tol. Inner products and norms are reduced over the processes on the host
// (MPI_Allreduce of j+1 numbers, as on the CPU); the small Hessenberg problem is solved on the host.
static int smoother_gmres_gpu_PRECISION( gmres_PRECISION_struct *p, level_struct *l ) {

  typedef std::complex<double> cd;
  interpolation_PRECISION_struct *is = &(l->is_PRECISION);
  operator_PRECISION_struct *op = &(l->oe_op_PRECISION);
  const int m = p->restart_length;
  const int n = op->num_even_sites * l->num_lattice_site_var;   // even half: the Schur system
  const int ld = l->inner_vector_size;                           // column stride of the basis V

  if ( smoother_cublas_PRECISION == NULL ) {
    CUBLAS_CHECK_INTERP( cublasCreate( &smoother_cublas_PRECISION ) );
    CUBLAS_CHECK_INTERP( cublasSetPointerMode( smoother_cublas_PRECISION, CUBLAS_POINTER_MODE_HOST ) );
  }
  cublasHandle_t h = smoother_cublas_PRECISION;
  if ( is->oe_V_cols < m+1 ) {
    if ( is->oe_V_gpu != NULL ) {
      CUDA_CHECK_INTERP( cudaFree( is->oe_V_gpu ) );
      CUDA_CHECK_INTERP( cudaFree( is->oe_h_gpu ) );
    }
    CUDA_CHECK_INTERP( cudaMalloc( (void**)&(is->oe_V_gpu), (size_t)(m+1)*ld*sizeof(cu_cmplx_PRECISION) ) );
    CUDA_CHECK_INTERP( cudaMalloc( (void**)&(is->oe_h_gpu), (size_t)(m+1)*sizeof(cu_cmplx_PRECISION) ) );
    is->oe_V_cols = m+1;
  }

  cu_cmplx_PRECISION *V = is->oe_V_gpu, *b = is->oe_b_gpu, *x = is->oe_x_gpu, *r = is->oe_chunk_gpu, *hd = is->oe_h_gpu;
  const cu_cmplx_PRECISION one = make_cu_cmplx_PRECISION(1,0), zero = make_cu_cmplx_PRECISION(0,0),
                           mone = make_cu_cmplx_PRECISION(-1,0);
  std::vector<cd> H( (size_t)m*(m+1) ), gam( m+1 ), sn( m ), cs( m ), y( m );
  std::vector<cu_cmplx_PRECISION> hloc( m+1 ), hglob( m+1 );
  auto Hc = [&]( int col, int row ) -> cd& { return H[ (size_t)col*(m+1) + row ]; };   // H[j][i] of the CPU code

  // global 2-norm over the even half of all processes
  auto global_norm = [&]( const cu_cmplx_PRECISION *v ) -> double {
    PRECISION nl = 0;
    CUBLAS_CHECK_INTERP( blas_nrm2( h, n, v, &nl ) );
    PRECISION sq = nl*nl, sq_glob = sq;
    if ( g.num_processes > 1 )
      MPI_Allreduce( &sq, &sq_glob, 1, MPI_PRECISION, MPI_SUM, g.comm_cart );
    return sqrt( (double)sq_glob );
  };

  double norm_r0 = 1, gamma_jp1 = 1;
  int finish = 0, iter = 0, j = -1;

  for ( int ol=0; ol<p->num_restart && finish==0; ol++ ) {
    int res;
    if ( ol == 0 && p->initial_guess_zero ) {
      res = _NO_RES;
      CUBLAS_CHECK_INTERP( blas_copy( h, n, b, r ) );                       // r = b
    } else {
      res = _RES;
      cuda_apply_schur_complement_PRECISION( r, x, op, l );                 // r = S x
      CUBLAS_CHECK_INTERP( blas_rscal( h, n, (PRECISION)(-1.0), r ) );      // r = b - S x
      CUBLAS_CHECK_INTERP( blas_axpy( h, n, &one, b, r ) );
    }
    double gamma0 = global_norm( r );
    gam[0] = gamma0;
    if ( ol == 0 ) norm_r0 = gamma0;
    CUBLAS_CHECK_INTERP( blas_copy( h, n, r, V ) );                         // v_0 = r / gamma_0
    CUBLAS_CHECK_INTERP( blas_rscal( h, n, (PRECISION)(1.0/gamma0), V ) );

    for ( int il=0; il<m && finish==0; il++ ) {
      j = il; iter++;
      cu_cmplx_PRECISION *vj = V + (size_t)j*ld, *w = V + (size_t)(j+1)*ld; // w is built in place of v_{j+1}

      cuda_apply_schur_complement_PRECISION( w, vj, op, l );                // w = S v_j

      // h_i = <v_i, w>, i = 0..j, reduced over the processes
      CUBLAS_CHECK_INTERP( blas_gemv( h, CUBLAS_OP_C, n, j+1, &one, V, ld, w, &zero, hd ) );
      CUDA_CHECK_INTERP( cudaMemcpy( hloc.data(), hd, (j+1)*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost ) );
      if ( g.num_processes > 1 )
        MPI_Allreduce( hloc.data(), hglob.data(), j+1, MPI_COMPLEX_PRECISION, MPI_SUM, g.comm_cart );
      else
        hglob = hloc;
      for ( int i=0; i<=j; i++ ) Hc(j,i) = cd( hglob[i].x, hglob[i].y );

      // w = w - sum_i h_i v_i  (classical Gram-Schmidt, as on the CPU)
      CUDA_CHECK_INTERP( cudaMemcpy( hd, hglob.data(), (j+1)*sizeof(cu_cmplx_PRECISION), cudaMemcpyHostToDevice ) );
      CUBLAS_CHECK_INTERP( blas_gemv( h, CUBLAS_OP_N, n, j+1, &mone, V, ld, hd, &one, w ) );

      // h_{j+1} = ||w||, v_{j+1} = w / h_{j+1}
      double hn = global_norm( w );
      Hc(j,j+1) = hn;
      if ( hn > 1e-15 ) CUBLAS_CHECK_INTERP( blas_rscal( h, n, (PRECISION)(1.0/hn), w ) );

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
          if ( gamma_jp1/norm_r0 > 1E+5 ) printf0("Divergence of smoother_gmres_gpu_PRECISION, iter = %d\n", iter );
        }
      } else {
        printf0("depth: %d, iter: %d, smoother GMRES (GPU) H(%d,%d) = %+lf+%lfi\n", l->depth, iter, j+1, j,
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
    CUDA_CHECK_INTERP( cudaMemcpy( hd, hglob.data(), (j+1)*sizeof(cu_cmplx_PRECISION), cudaMemcpyHostToDevice ) );
    int add = ( res == _NO_RES ) ? ol : 1;
    CUBLAS_CHECK_INTERP( blas_gemv( h, CUBLAS_OP_N, n, j+1, &one, V, ld, hd, add ? &one : &zero, x ) );
  }
  CUDA_CHECK_INTERP( cudaDeviceSynchronize() );
  return iter;
}


// Depth-0 method-4 smoother after the GPU residual, entirely on the GPU: odd-even preparation,
// GMRES on the Schur complement, reconstruction, phi += correction, one copy of phi to the host.
// Called by all threads (the work is done by the master); returns the number of GMRES iterations.
extern "C" int cuda_smoother_oe_gmres_PRECISION( vector_PRECISION phi, gmres_PRECISION_struct *p, level_struct *l,
                                                 struct Thread *threading ) {

  int iter = 0;
  SYNC_CORES(threading)
  SYNC_HYPERTHREADS(threading)
  START_LOCKED_MASTER(threading)
  smoother_oe_buffers_PRECISION( l );
  smoother_oe_prepare_core_PRECISION( l );
  iter = smoother_gmres_gpu_PRECISION( p, l );
  smoother_oe_finish_core_PRECISION( phi, l );
  END_LOCKED_MASTER(threading)
  SYNC_HYPERTHREADS(threading)
  return iter;
}

#endif
