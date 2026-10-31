#include <mpi.h>
#include <stdio.h>

extern "C"{

  #define IMPORT_FROM_EXTERN_C
  #include "main.h"
  #undef IMPORT_FROM_EXTERN_C

}

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
__device__ __forceinline__ void restrict_op_entry_PRECISION( const cu_cmplx_PRECISION *op, size_t row, int j, int nev,
                                                             size_t vector_size, PRECISION *re, PRECISION *im ) {
#if defined( SSE ) && defined( INTERPOLATION_OPERATOR_LAYOUT_OPTIMIZED_PRECISION )
  // SSE layout: test vectors in blocks of S = SIMD_LENGTH_PRECISION, block jb starts at jb*S*vector_size;
  // each row stores [re_0 .. re_{S-1}, im_0 .. im_{S-1}]
  const int S = SIMD_LENGTH_PRECISION;
  const PRECISION *opf = (const PRECISION*) op;
  size_t base = 2*( (size_t)(j/S)*S*vector_size + (size_t)S*row );
  *re = opf[ base + j%S ];
  *im = opf[ base + S + j%S ];
#else
  // plain layout: interleaved complex numbers, test vector index fastest
  cu_cmplx_PRECISION o = op[ row*nev + j ];
  *re = cu_creal_PRECISION(o);
  *im = cu_cimag_PRECISION(o);
#endif
}


// one thread per coarse entry phi_c[ i*2*nev + k1*nev + j ]
__global__ void cuda_restrict_PRECISION_kernel( cuda_vector_PRECISION phi_c, const cu_cmplx_PRECISION *phi,
                                                const cu_cmplx_PRECISION *op, int num_agg, int nev, int npev,
                                                int agg_sites, size_t vector_size ) {

  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if ( idx >= num_agg*2*nev ) return;

  int j  = idx % nev;
  int k1 = (idx / nev) % 2;
  int i  = idx / (2*nev);

  PRECISION re = 0, im = 0;

  // same summation order as the CPU code: sites k, then parent variables k2
  for ( int k=0; k<agg_sites; k++ ) {
    size_t row = ((size_t)i*agg_sites + k)*2*npev + (size_t)k1*npev;
    for ( int k2=0; k2<npev; k2++, row++ ) {
      cu_cmplx_PRECISION p = phi[row];
      PRECISION p_re = cu_creal_PRECISION(p), p_im = cu_cimag_PRECISION(p), o_re, o_im;
      restrict_op_entry_PRECISION( op, row, j, nev, vector_size, &o_re, &o_im );
      // phi * conj(op)
      re += p_re*o_re + p_im*o_im;
      im += p_im*o_re - p_re*o_im;
    }
  }

  phi_c[idx] = make_cu_cmplx_PRECISION( re, im );
}


// copies l->is_PRECISION.op to the GPU (allocates device buffers at first call)
// to be called by one thread only, after the operator has been (re)defined
extern "C" void cuda_interpolation_PRECISION_op_to_gpu( level_struct *l ) {

  size_t op_size  = interpolation_op_size_PRECISION( l ),
         n_fine   = (size_t)l->inner_vector_size,
         n_coarse = (size_t)l->is_PRECISION.num_agg*2*l->num_eig_vect;

  if ( l->is_PRECISION.op_gpu == NULL ) {
    CUDA_CHECK_INTERP( cudaMalloc( (void**)&(l->is_PRECISION.op_gpu),    op_size*sizeof(cu_cmplx_PRECISION) ) );
    CUDA_CHECK_INTERP( cudaMalloc( (void**)&(l->is_PRECISION.phi_gpu),   n_fine*sizeof(cu_cmplx_PRECISION) ) );
    CUDA_CHECK_INTERP( cudaMalloc( (void**)&(l->is_PRECISION.phi_c_gpu), n_coarse*sizeof(cu_cmplx_PRECISION) ) );
  }

  CUDA_CHECK_INTERP( cudaMemcpy( l->is_PRECISION.op_gpu, l->is_PRECISION.op,
                                 op_size*sizeof(cu_cmplx_PRECISION), cudaMemcpyHostToDevice ) );
}


extern "C" void cuda_interpolation_PRECISION_free( level_struct *l ) {

  if ( l->is_PRECISION.op_gpu != NULL ) {
    CUDA_CHECK_INTERP( cudaFree( l->is_PRECISION.op_gpu ) );
    CUDA_CHECK_INTERP( cudaFree( l->is_PRECISION.phi_gpu ) );
    CUDA_CHECK_INTERP( cudaFree( l->is_PRECISION.phi_c_gpu ) );
    l->is_PRECISION.op_gpu    = NULL;
    l->is_PRECISION.phi_gpu   = NULL;
    l->is_PRECISION.phi_c_gpu = NULL;
  }
}


// GPU version of restrict_PRECISION: same interface, phi and phi_c are host vectors
extern "C" void cuda_restrict_PRECISION( vector_PRECISION phi_c, vector_PRECISION phi, level_struct *l, struct Thread *threading ) {

  SYNC_CORES(threading)
  SYNC_HYPERTHREADS(threading)

  PROF_PRECISION_START( _PR, threading );

  START_LOCKED_MASTER(threading)

  int num_agg   = l->is_PRECISION.num_agg,
      nev       = l->num_eig_vect,
      npev      = l->num_lattice_site_var/2,
      agg_sites = l->num_inner_lattice_sites / num_agg;
  size_t n_fine   = (size_t)l->inner_vector_size,
         n_coarse = (size_t)num_agg*2*nev;

  if ( l->is_PRECISION.op_gpu == NULL ) {
    fprintf( stderr, "cuda_restrict_PRECISION: interpolation operator not on GPU (depth=%d)\n", l->depth );
    MPI_Abort( MPI_COMM_WORLD, 1 );
  }

  CUDA_CHECK_INTERP( cudaMemcpy( l->is_PRECISION.phi_gpu, phi, n_fine*sizeof(cu_cmplx_PRECISION),
                                 cudaMemcpyHostToDevice ) );

  int threads_per_block = 128;
  int nr_blocks = (int)( (n_coarse + threads_per_block - 1) / threads_per_block );
  cuda_restrict_PRECISION_kernel<<< nr_blocks, threads_per_block >>>( l->is_PRECISION.phi_c_gpu, l->is_PRECISION.phi_gpu,
                                                                      l->is_PRECISION.op_gpu, num_agg, nev, npev,
                                                                      agg_sites, (size_t)l->vector_size );
  CUDA_CHECK_INTERP( cudaGetLastError() );

  // synchronous copy on the default stream: waits for the kernel to finish
  CUDA_CHECK_INTERP( cudaMemcpy( l->next_level->gs_PRECISION.transfer_buffer, l->is_PRECISION.phi_c_gpu,
                                 n_coarse*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost ) );

  // distribution to the coarse-level processes stays on the CPU (MPI)
  vector_PRECISION_gather( phi_c, l->next_level->gs_PRECISION.transfer_buffer, l->next_level );

  END_LOCKED_MASTER(threading)

  PROF_PRECISION_STOP( _PR, 1, threading );
}

#endif
