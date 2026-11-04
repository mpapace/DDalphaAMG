#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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


// Dedicated non-blocking stream for the restriction (created at first use, kept for the
// whole run). Copies and kernel launched on the legacy default stream synchronise implicitly
// with all the other (blocking) streams of the fork; on a non-blocking stream they do not.
// No implicit synchronisation is needed here: phi is a host vector, and the operator is
// copied with a synchronous cudaMemcpy before the first restriction.
// Compile with -DCUDA_RESTRICT_DEFAULT_STREAM to use the legacy default stream as before.
static cudaStream_t restrict_stream_PRECISION( void ) {
#ifdef CUDA_RESTRICT_DEFAULT_STREAM
  return 0;
#else
  static cudaStream_t s = 0;
  static int created = 0;
  if ( !created ) {
    CUDA_CHECK_INTERP( cudaStreamCreateWithFlags( &s, cudaStreamNonBlocking ) );
    created = 1;
  }
  return s;
#endif
}


// ---------------------------------------------------------------------------------------
// Pinned (page-locked) host memory for the restriction copies.
// cudaMemcpy from/to pageable memory (plain malloc) needs extra driver work at every call
// (~20 us for the 1.5 MiB copy at depth 1). Host buffers that live as long as the level
// are registered once with cudaHostRegister and unregistered in
// cuda_interpolation_PRECISION_free, which the vanilla code calls before it frees any of them.
// Registration costs tens of ms per GB, so only the coarse levels (depth >= 1) are pinned:
// there the buffers are small (MB) and the pageable overhead is large compared with the copy.
// At depth 0 (copies of ~100 MB, overhead ~15 us out of ~250 us) it would not pay off.
// Pinned at depth >= 1:
//   - transfer buffer of the next level (destination of the DtoH copy)
//   - test vectors of the level (phi during the setup)
//   - l->vbuf_PRECISION[3]          (phi in the vcycle, residual after smoothing)
//   - basis vectors V of the FGMRES of the level (phi in the K-cycle), registered at the
//     first restriction call, because they may be allocated after the operator setup;
//     they are freed by next_level_PRECISION_free after cuda_interpolation_PRECISION_free
// Any other phi is copied from pageable memory as before; such calls are counted and listed
// in the report printed by cuda_interpolation_PRECISION_free (rank 0).
// Compile with -DCUDA_RESTRICT_NO_PIN to disable the registration (the report stays).
// ---------------------------------------------------------------------------------------

#define RESTRICT_MAX_PINNED_PRECISION 64
#define RESTRICT_MAX_PHI_PRECISION   256
#define RESTRICT_MAX_DEPTH_PRECISION  64

typedef struct {
  void *ptr;
  size_t bytes;
  int depth;
  int owned;          // 1: registered by us (unregister at free), 0: was already pinned
  const char *name;
} restrict_pinned_region_PRECISION;

typedef struct {
  const void *ptr;
  int depth;
  long calls;
  const char *label;
  int index;
} restrict_phi_stat_PRECISION;

static restrict_pinned_region_PRECISION restrict_pinned_PRECISION[RESTRICT_MAX_PINNED_PRECISION];
static int restrict_num_pinned_PRECISION = 0;
static restrict_phi_stat_PRECISION restrict_phi_stats_PRECISION[RESTRICT_MAX_PHI_PRECISION];
static int restrict_num_phi_stats_PRECISION = 0;
static long restrict_calls_pinned_PRECISION[RESTRICT_MAX_DEPTH_PRECISION]   = {0};
static long restrict_calls_pageable_PRECISION[RESTRICT_MAX_DEPTH_PRECISION] = {0};

static int restrict_rank_PRECISION( void ) {
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  return rank;
}

// registers [ptr, ptr+bytes) as pinned memory (once per pointer)
static void restrict_pin_PRECISION( void *ptr, size_t bytes, int depth, const char *name ) {
#ifndef CUDA_RESTRICT_NO_PIN
  if ( ptr == NULL || bytes == 0 ) return;
  for ( int r=0; r<restrict_num_pinned_PRECISION; r++ )
    if ( restrict_pinned_PRECISION[r].ptr == ptr ) return;
  if ( restrict_num_pinned_PRECISION >= RESTRICT_MAX_PINNED_PRECISION ) {
    fprintf( stderr, "cuda_restrict_PRECISION: too many pinned regions, %s stays pageable\n", name );
    return;
  }

  int owned = 1;
  cudaError_t err = cudaHostRegister( ptr, bytes, cudaHostRegisterDefault );
  if ( err == cudaErrorHostMemoryAlreadyRegistered ) {
    cudaGetLastError();   // clear the error: the memory is already pinned, not ours to unregister
    owned = 0;
  } else if ( err != cudaSuccess ) {
    cudaGetLastError();
    fprintf( stderr, "cuda_restrict_PRECISION: cudaHostRegister of %s (%zu bytes) failed: %s, stays pageable\n",
             name, bytes, cudaGetErrorString(err) );
    return;
  }

  restrict_pinned_region_PRECISION *reg = &restrict_pinned_PRECISION[restrict_num_pinned_PRECISION++];
  reg->ptr = ptr; reg->bytes = bytes; reg->depth = depth; reg->owned = owned; reg->name = name;

  if ( restrict_rank_PRECISION() == 0 )
    printf( "[restrict pinned] depth %d: %-24s %10.3f MiB pinned%s\n", depth, name,
            bytes/(1024.0*1024.0), owned ? "" : " (already registered)" );
#endif
}

// unregisters all regions of one level
static void restrict_unpin_level_PRECISION( int depth ) {
  int keep = 0;
  for ( int r=0; r<restrict_num_pinned_PRECISION; r++ ) {
    restrict_pinned_region_PRECISION *reg = &restrict_pinned_PRECISION[r];
    if ( reg->depth == depth ) {
      if ( reg->owned ) {
        cudaError_t err = cudaHostUnregister( reg->ptr );
        if ( err != cudaSuccess ) {
          cudaGetLastError();
          fprintf( stderr, "cuda_restrict_PRECISION: cudaHostUnregister of %s failed: %s\n",
                   reg->name, cudaGetErrorString(err) );
        }
      }
    } else {
      restrict_pinned_PRECISION[keep++] = *reg;
    }
  }
  restrict_num_pinned_PRECISION = keep;
}

// 1 if [ptr, ptr+bytes) lies inside a pinned region
static int restrict_is_pinned_PRECISION( const void *ptr, size_t bytes ) {
  const char *p = (const char*) ptr;
  for ( int r=0; r<restrict_num_pinned_PRECISION; r++ ) {
    const char *b = (const char*) restrict_pinned_PRECISION[r].ptr;
    if ( p >= b && p + bytes <= b + restrict_pinned_PRECISION[r].bytes ) return 1;
  }
  return 0;
}

// name of phi among the known vectors of the level (for the report)
static const char *restrict_phi_label_PRECISION( const void *phi, level_struct *l, int *index ) {
  gmres_PRECISION_struct *p = &(l->p_PRECISION);
  *index = -1;
  if ( phi == p->b ) return "l->p_PRECISION.b";
  if ( phi == p->x ) return "l->p_PRECISION.x";
  if ( phi == p->r ) return "l->p_PRECISION.r";
  if ( phi == p->w ) return "l->p_PRECISION.w";
  if ( p->V != NULL )
    for ( int i=0; i<=p->restart_length; i++ ) if ( phi == p->V[i] ) { *index = i; return "l->p_PRECISION.V"; }
  if ( p->Z != NULL )
    for ( int i=0; i<p->restart_length; i++ ) if ( phi == p->Z[i] ) { *index = i; return "l->p_PRECISION.Z"; }
  for ( int i=0; i<9; i++ ) if ( phi == l->vbuf_PRECISION[i] ) { *index = i; return "l->vbuf_PRECISION"; }
  return "unknown";
}

// counts one restriction call, by kind of host memory of phi
static void restrict_note_phi_PRECISION( const void *phi, size_t bytes, level_struct *l ) {
  int d = ( l->depth < RESTRICT_MAX_DEPTH_PRECISION ) ? l->depth : RESTRICT_MAX_DEPTH_PRECISION-1;
  if ( restrict_is_pinned_PRECISION( phi, bytes ) ) {
    restrict_calls_pinned_PRECISION[d]++;
    return;
  }
  restrict_calls_pageable_PRECISION[d]++;
  for ( int s=0; s<restrict_num_phi_stats_PRECISION; s++ )
    if ( restrict_phi_stats_PRECISION[s].ptr == phi && restrict_phi_stats_PRECISION[s].depth == l->depth ) {
      restrict_phi_stats_PRECISION[s].calls++;
      return;
    }
  if ( restrict_num_phi_stats_PRECISION < RESTRICT_MAX_PHI_PRECISION ) {
    restrict_phi_stat_PRECISION *st = &restrict_phi_stats_PRECISION[restrict_num_phi_stats_PRECISION++];
    st->ptr = phi; st->depth = l->depth; st->calls = 1;
    st->label = restrict_phi_label_PRECISION( phi, l, &(st->index) );
  }
}

// pins the basis vectors V[0..restart_length] of the FGMRES of the level (depth >= 1 only).
// Called at every restriction call; does nothing once the block is registered or if the
// level has no FGMRES. V is registered as one block if contiguous, else vector by vector.
static void restrict_pin_fgmres_PRECISION( level_struct *l ) {
  if ( l->depth < 1 ) return;
  gmres_PRECISION_struct *p = &(l->p_PRECISION);
  if ( p->V == NULL || p->V[0] == NULL || p->restart_length <= 0 ) return;
  size_t vbytes = (size_t)l->vector_size * sizeof(cu_cmplx_PRECISION);
  int contiguous = 1;
  for ( int i=1; i<=p->restart_length; i++ )
    if ( p->V[i] != p->V[0] + (size_t)i*l->vector_size ) { contiguous = 0; break; }
  if ( contiguous ) {
    restrict_pin_PRECISION( p->V[0], (size_t)(p->restart_length+1)*vbytes, l->depth, "FGMRES V (block)" );
  } else {
    for ( int i=0; i<=p->restart_length; i++ )
      restrict_pin_PRECISION( p->V[i], vbytes, l->depth, "FGMRES V[i]" );
  }
}

// report of one level (rank 0), then its statistics are reset
static void restrict_report_level_PRECISION( int depth ) {
  int d = ( depth < RESTRICT_MAX_DEPTH_PRECISION ) ? depth : RESTRICT_MAX_DEPTH_PRECISION-1;
  if ( restrict_rank_PRECISION() == 0 &&
       ( restrict_calls_pinned_PRECISION[d] + restrict_calls_pageable_PRECISION[d] ) > 0 ) {
    printf( "[restrict pinned] depth %d: %ld calls with phi in pinned memory, %ld in pageable memory\n",
            depth, restrict_calls_pinned_PRECISION[d], restrict_calls_pageable_PRECISION[d] );
    for ( int s=0; s<restrict_num_phi_stats_PRECISION; s++ ) {
      restrict_phi_stat_PRECISION *st = &restrict_phi_stats_PRECISION[s];
      if ( st->depth != depth ) continue;
      if ( st->index >= 0 )
        printf( "[restrict pinned]   pageable phi %p: %6ld calls  (%s[%d])\n", st->ptr, st->calls, st->label, st->index );
      else
        printf( "[restrict pinned]   pageable phi %p: %6ld calls  (%s)\n", st->ptr, st->calls, st->label );
    }
    fflush( stdout );
  }
  restrict_calls_pinned_PRECISION[d] = 0;
  restrict_calls_pageable_PRECISION[d] = 0;
  int keep = 0;
  for ( int s=0; s<restrict_num_phi_stats_PRECISION; s++ )
    if ( restrict_phi_stats_PRECISION[s].depth != depth )
      restrict_phi_stats_PRECISION[keep++] = restrict_phi_stats_PRECISION[s];
  restrict_num_phi_stats_PRECISION = keep;
}


// launches the restriction kernel for the given operator layout
template<bool SSE_LAYOUT>
static void cuda_restrict_PRECISION_launch( cuda_vector_PRECISION phi_c_gpu, const cu_cmplx_PRECISION *phi_gpu,
                                            const cu_cmplx_PRECISION *op_gpu, level_struct *l,
                                            cudaStream_t stream ) {

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

  cuda_restrict_PRECISION_kernel<SSE_LAYOUT><<< nr_blocks, block, smem, stream >>>( phi_c_gpu, phi_gpu, op_gpu, nev, npev,
                                                                            agg_sites, (size_t)l->vector_size,
                                                                            restrict_simd_length_PRECISION() );
  CUDA_CHECK_INTERP( cudaGetLastError() );
}


#ifdef CUDA_RESTRICT_TEST

// one test per level (indexed by depth)
static int restrict_tested_PRECISION[64] = {0};

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

// Unit test of the plain-layout kernel (the one used in builds without SSE):
//   1. converts the host operator to the plain layout of interpolation_generic.c
//   2. computes the restriction on the CPU with the loop of interpolation_generic.c
//   3. runs the plain-layout instance of the GPU kernel
//   4. compares both GPU results (plain-layout test and production) with the CPU reference
// Expects phi already uploaded to l->is_PRECISION.phi_gpu and the production result in the transfer buffer.
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
  cuda_restrict_PRECISION_launch<false>( phi_c_test_gpu, l->is_PRECISION.phi_gpu, op_plain_gpu, l, 0 );
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

    // host buffers of the restriction copies that live as long as the level (see above);
    // only on the coarse levels, where pinning pays off
    if ( l->depth >= 1 ) {
      restrict_pin_PRECISION( l->next_level->gs_PRECISION.transfer_buffer,
                              (size_t)l->next_level->gs_PRECISION.dist_inner_lattice_sites
                              * (size_t)l->next_level->num_lattice_site_var * sizeof(cu_cmplx_PRECISION),
                              l->depth, "next level transfer_buffer" );
      if ( l->is_PRECISION.test_vector != NULL )
        restrict_pin_PRECISION( l->is_PRECISION.test_vector[0],
                                (size_t)l->num_eig_vect * n_fine * sizeof(cu_cmplx_PRECISION),
                                l->depth, "test vectors" );
      restrict_pin_PRECISION( l->vbuf_PRECISION[3], (size_t)l->vector_size * sizeof(cu_cmplx_PRECISION),
                              l->depth, "vbuf_PRECISION[3]" );
    }
  }

  CUDA_CHECK_INTERP( cudaMemcpy( l->is_PRECISION.op_gpu, l->is_PRECISION.op,
                                 op_size*sizeof(cu_cmplx_PRECISION), cudaMemcpyHostToDevice ) );
}


extern "C" void cuda_interpolation_PRECISION_free( level_struct *l ) {

  // called by the vanilla code before it frees the host buffers of the level
  restrict_report_level_PRECISION( l->depth );
  restrict_unpin_level_PRECISION( l->depth );

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

  size_t n_fine   = (size_t)l->inner_vector_size,
         n_coarse = (size_t)l->is_PRECISION.num_agg*2*l->num_eig_vect;

  if ( l->is_PRECISION.op_gpu == NULL ) {
    fprintf( stderr, "cuda_restrict_PRECISION: interpolation operator not on GPU (depth=%d)\n", l->depth );
    MPI_Abort( MPI_COMM_WORLD, 1 );
  }

  restrict_pin_fgmres_PRECISION( l );
  restrict_note_phi_PRECISION( phi, n_fine*sizeof(cu_cmplx_PRECISION), l );

  // copy, kernel, copy back on the restriction stream, one synchronisation at the end
  cudaStream_t stream = restrict_stream_PRECISION();
  CUDA_CHECK_INTERP( cudaMemcpyAsync( l->is_PRECISION.phi_gpu, phi, n_fine*sizeof(cu_cmplx_PRECISION),
                                      cudaMemcpyHostToDevice, stream ) );

  cuda_restrict_PRECISION_launch<restrict_sse_layout_PRECISION>( l->is_PRECISION.phi_c_gpu, l->is_PRECISION.phi_gpu,
                                                                 l->is_PRECISION.op_gpu, l, stream );

  CUDA_CHECK_INTERP( cudaMemcpyAsync( l->next_level->gs_PRECISION.transfer_buffer, l->is_PRECISION.phi_c_gpu,
                                      n_coarse*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost, stream ) );
  CUDA_CHECK_INTERP( cudaStreamSynchronize( stream ) );

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

#endif
