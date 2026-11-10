// Outer FGMRES of the solve on the GPU (step E of the finest-level GPU residency).
//
// Same algorithm as fgmres_double for the outer solver (right preconditioning with the float
// V-cycle, classical Gram-Schmidt, restarts, stop on the estimated relative residual), with the
// fine vectors kept on the GPU: V, Z, x, b in double, lexicographic ordering of g.op_double,
// 12 components per site. The V-cycle reads its right-hand side from the GPU (eta_gpu of the
// depth-0 interpolation) and leaves its result there (phi_gpu), see cuda_interpolation_generic.cu.
// Only scalars go through the host (MPI reductions, Hessenberg matrix, Givens rotations).
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

  // vcycle_float.h is not included by main.h in CUDA files (IMPORT_FROM_EXTERN_C)
  void vcycle_float( vector_float phi, vector_float Dphi, vector_float eta,
                     int res, level_struct *l, struct Thread *threading );
}

#include "cuda_componentwise.h"
#include "cuda_outer_fgmres.h"

#ifdef CUDA_OPT

#define CUDA_CHECK_OUTER(call)                                                    \
  do {                                                                            \
    cudaError_t err_ = (call);                                                    \
    if ( err_ != cudaSuccess ) {                                                  \
      fprintf( stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,           \
               cudaGetErrorString(err_) );                                        \
      MPI_Abort( MPI_COMM_WORLD, 1 );                                             \
    }                                                                             \
  } while (0)

#define CUBLAS_CHECK_OUTER(call)                                                  \
  do {                                                                            \
    cublasStatus_t st_ = (call);                                                  \
    if ( st_ != CUBLAS_STATUS_SUCCESS ) {                                         \
      fprintf( stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__, __LINE__ ); \
      MPI_Abort( MPI_COMM_WORLD, 1 );                                             \
    }                                                                             \
  } while (0)

typedef std::complex<double> cd_outer;

// state of the outer solver: written by the master, read by all threads after a barrier
static struct {
  cublasHandle_t h;
  cu_cmplx_double *V, *Z, *x, *b, *w, *hd;   // V: m+1 columns, Z: m columns, column stride vl
  int *tt;                                     // lexicographic site -> Schwarz site (s_float)
  size_t vl;
  int m;
  int finish, stop_restart;
} og = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0, 0, 0, 0 };


// double, lexicographic -> float, Schwarz ordering (trans_float on the GPU)
__global__ void outer_trans_float_kernel( cu_cmplx_float *out, const cu_cmplx_double *in, const int *tt,
                                          size_t n_sites, int nsv ) {
  size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
  if ( i >= n_sites*nsv ) return;
  size_t site = i / nsv, c = i % nsv;
  cu_cmplx_double v = in[i];
  out[(size_t)tt[site]*nsv + c] = make_cuFloatComplex( (float)v.x, (float)v.y );
}

// float, Schwarz ordering -> double, lexicographic (trans_back_float on the GPU)
__global__ void outer_trans_back_float_kernel( cu_cmplx_double *out, const cu_cmplx_float *in, const int *tt,
                                               size_t n_sites, int nsv ) {
  size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
  if ( i >= n_sites*nsv ) return;
  size_t site = i / nsv, c = i % nsv;
  cu_cmplx_float v = in[(size_t)tt[site]*nsv + c];
  out[i] = make_cuDoubleComplex( (double)v.x, (double)v.y );
}


extern "C" int cuda_outer_fgmres_applicable( gmres_double_struct *p, level_struct *l ) {
#if defined( OUTER_FGMRES_ON_CPU ) || defined( GCR_SMOOTHER ) || defined( RICHARDSON_SMOOTHER ) || \
    defined( CUDA_RESTRICT_TEST ) || defined( HALF_PREC_STORAGE )
  return 0;
#else
  const double *sh = (const double*) &(p->shift);
  return ( l->depth == 0 && g.mixed_precision == 1 && g.method == 4 && g.odd_even && g.interpolation &&
           l->level > 0 && l->n_cy == 1 && p->preconditioner != NULL && p->kind == _RIGHT &&
           sh[0] == 0 && sh[1] == 0 && p->op == &(g.op_double) && g.vt.p_end == NULL && !g.vt.evaluation &&
           p->op->x_componentwise_gpu != NULL && p->op->w_componentwise_gpu != NULL &&
           l->is_float.op_gpu != NULL && l->is_float.eta_gpu != NULL && l->is_float.phi_gpu != NULL &&
           l->num_lattice_site_var == 12 );
#endif
}


static void outer_buffers( gmres_double_struct *p, level_struct *l ) {
  size_t vl = (size_t)l->inner_vector_size;
  int m = p->restart_length;
  if ( og.h == NULL ) {
    CUBLAS_CHECK_OUTER( cublasCreate( &og.h ) );
    CUBLAS_CHECK_OUTER( cublasSetPointerMode( og.h, CUBLAS_POINTER_MODE_HOST ) );
  }
  if ( og.vl != vl || og.m != m ) {
    if ( og.V != NULL ) {
      CUDA_CHECK_OUTER( cudaFree( og.V ) ); CUDA_CHECK_OUTER( cudaFree( og.Z ) );
      CUDA_CHECK_OUTER( cudaFree( og.x ) ); CUDA_CHECK_OUTER( cudaFree( og.b ) );
      CUDA_CHECK_OUTER( cudaFree( og.w ) ); CUDA_CHECK_OUTER( cudaFree( og.hd ) );
    }
    CUDA_CHECK_OUTER( cudaMalloc( (void**)&og.V, (size_t)(m+1)*vl*sizeof(cu_cmplx_double) ) );
    CUDA_CHECK_OUTER( cudaMalloc( (void**)&og.Z, (size_t)m*vl*sizeof(cu_cmplx_double) ) );
    CUDA_CHECK_OUTER( cudaMalloc( (void**)&og.x, vl*sizeof(cu_cmplx_double) ) );
    CUDA_CHECK_OUTER( cudaMalloc( (void**)&og.b, vl*sizeof(cu_cmplx_double) ) );
    CUDA_CHECK_OUTER( cudaMalloc( (void**)&og.w, vl*sizeof(cu_cmplx_double) ) );
    CUDA_CHECK_OUTER( cudaMalloc( (void**)&og.hd, (size_t)(m+1)*sizeof(cu_cmplx_double) ) );
    og.vl = vl; og.m = m;
  }
  if ( og.tt == NULL ) {
    // the translation table is fixed after the setup of the Schwarz layout
    size_t n = (size_t)l->num_inner_lattice_sites;
    CUDA_CHECK_OUTER( cudaMalloc( (void**)&og.tt, n*sizeof(int) ) );
    CUDA_CHECK_OUTER( cudaMemcpy( og.tt, l->s_float.op.translation_table, n*sizeof(int), cudaMemcpyHostToDevice ) );
  }
}

// out = D in, D = g.op_double on the GPU (as the d_plus_clover proxy, without the host copies)
static void outer_apply_D( cu_cmplx_double *out, const cu_cmplx_double *in, operator_double_struct *op,
                           level_struct *l, struct Thread *threading ) {
  const size_t n = (size_t)l->num_inner_lattice_sites, nsv = (size_t)l->num_lattice_site_var;
  reorderArrayByComponent<<< minGridSizeForN( n, 128 ), 128 >>>( op->x_componentwise_gpu, in, nsv, n );
  CUDA_CHECK_OUTER( cudaDeviceSynchronize() );
  cuda_d_plus_clover_double( op->w_componentwise_gpu, op->x_componentwise_gpu, op, l, threading );
  reorderArrayByChunks<<< minGridSizeForN( n, 128 ), 128 >>>( out, op->w_componentwise_gpu, nsv, n );
  CUDA_CHECK_OUTER( cudaDeviceSynchronize() );
}

// global 2-norm (sum of the local squares, MPI_Allreduce, sqrt: as global_norm_double)
static double outer_norm( const cu_cmplx_double *v, size_t vl ) {
  double nl = 0;
  CUBLAS_CHECK_OUTER( cublasDznrm2( og.h, (int)vl, v, 1, &nl ) );
  double sq = nl*nl, sq_glob = sq;
  if ( g.num_processes > 1 )
    MPI_Allreduce( &sq, &sq_glob, 1, MPI_DOUBLE, MPI_SUM, g.comm_cart );
  return sqrt( sq_glob );
}


extern "C" int cuda_outer_fgmres_double( gmres_double_struct *p, level_struct *l, struct Thread *threading ) {

  // master-only state (the other threads only take part in the V-cycle)
  static std::vector<cd_outer> H, gam, sn, cs, y;
  static std::vector<cu_cmplx_double> hloc, hglob;
  static double norm_r0, gamma_jp1, t0, t1;
  static int iter, j, res;

  const int m = p->restart_length;
  const size_t vl = (size_t)l->inner_vector_size, n_sites = (size_t)l->num_inner_lattice_sites;
  const int nsv = l->num_lattice_site_var;
  const cu_cmplx_double one = make_cuDoubleComplex(1,0), zero = make_cuDoubleComplex(0,0),
                        mone = make_cuDoubleComplex(-1,0);
  const double minus1 = -1.0;
  auto Hc = [&]( int col, int row ) -> cd_outer& { return H[ (size_t)col*(m+1) + row ]; };   // H[j][i] of fgmres

  START_LOCKED_MASTER(threading)
  if ( p->timing || p->print ) prof_init( l );
  t0 = MPI_Wtime();
#if defined(TRACK_RES) && !defined(WILSON_BENCHMARK)
  if ( p->print && g.print > 0 ) printf0("+----------------------------------------------------------+\n");
#endif
  outer_buffers( p, l );
  H.assign( (size_t)m*(m+1), 0 ); gam.assign( m+1, 0 ); sn.assign( m, 0 ); cs.assign( m, 0 ); y.assign( m, 0 );
  hloc.resize( m+1 ); hglob.resize( m+1 );
  norm_r0 = 1; gamma_jp1 = 1; iter = 0; j = -1;
  og.finish = 0;
  // the only host-to-device copies of fine vectors in the solve
  CUDA_CHECK_OUTER( cudaMemcpy( og.b, p->b, vl*sizeof(cu_cmplx_double), cudaMemcpyHostToDevice ) );
  if ( !p->initial_guess_zero )
    CUDA_CHECK_OUTER( cudaMemcpy( og.x, p->x, vl*sizeof(cu_cmplx_double), cudaMemcpyHostToDevice ) );
  END_LOCKED_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)

  for ( int ol=0; ol<p->num_restart && og.finish==0; ol++ ) {

    START_LOCKED_MASTER(threading)
    if ( ol == 0 && p->initial_guess_zero ) {
      res = _NO_RES;
      CUBLAS_CHECK_OUTER( cublasZcopy( og.h, (int)vl, og.b, 1, og.w, 1 ) );            // r = b
    } else {
      res = _RES;
      outer_apply_D( og.w, og.x, p->op, l, threading );                                  // w = D x
      CUBLAS_CHECK_OUTER( cublasZdscal( og.h, (int)vl, &minus1, og.w, 1 ) );
      CUBLAS_CHECK_OUTER( cublasZaxpy( og.h, (int)vl, &one, og.b, 1, og.w, 1 ) );       // r = b - D x
    }
    double gamma0 = outer_norm( og.w, vl );
    gam[0] = gamma0;
    if ( ol == 0 ) norm_r0 = gamma0;
    CUBLAS_CHECK_OUTER( cublasZcopy( og.h, (int)vl, og.w, 1, og.V, 1 ) );                // v_0 = r / gamma_0
    double s0 = 1.0/gamma0;
    CUBLAS_CHECK_OUTER( cublasZdscal( og.h, (int)vl, &s0, og.V, 1 ) );
    og.stop_restart = 0;
    END_LOCKED_MASTER(threading)
    SYNC_MASTER_TO_ALL(threading)

    for ( int il=0; il<m && og.finish==0 && og.stop_restart==0; il++ ) {

      // Z_j = M V_j: V_j to the depth-0 interpolation buffer eta_gpu (float, Schwarz ordering)
      START_LOCKED_MASTER(threading)
      j = il; iter++;
      interpolation_float_struct *is = &(l->is_float);
      outer_trans_float_kernel<<< minGridSizeForN( vl, 256 ), 256 >>>( is->eta_gpu, og.V + (size_t)j*vl, og.tt,
                                                                       n_sites, nsv );
      CUDA_CHECK_OUTER( cudaDeviceSynchronize() );
      is->eta_resident_host = l->sbuf_float[0];
      is->keep_phi_result = 1;
      is->phi_result_host = NULL;
      END_LOCKED_MASTER(threading)
      SYNC_MASTER_TO_ALL(threading)

      vcycle_float( l->sbuf_float[1], NULL, l->sbuf_float[0], _NO_RES, l, threading );

      START_LOCKED_MASTER(threading)
      interpolation_float_struct *is = &(l->is_float);
      if ( is->phi_result_host != l->sbuf_float[1] ) {
        // the smoother did not leave its result on the GPU (CPU fallback): take it from the host
        CUDA_CHECK_OUTER( cudaMemcpy( is->phi_gpu, l->sbuf_float[1], vl*sizeof(cu_cmplx_float),
                                      cudaMemcpyHostToDevice ) );
      }
      is->keep_phi_result = 0; is->phi_result_host = NULL;
      is->eta_resident_host = NULL; is->eta_stale_host = NULL;
      cu_cmplx_double *zj = og.Z + (size_t)j*vl, *w = og.V + (size_t)(j+1)*vl;
      outer_trans_back_float_kernel<<< minGridSizeForN( vl, 256 ), 256 >>>( zj, is->phi_gpu, og.tt, n_sites, nsv );
      CUDA_CHECK_OUTER( cudaDeviceSynchronize() );

      outer_apply_D( w, zj, p->op, l, threading );                                       // w = D Z_j

      // h_i = <v_i, w>, i = 0..j, reduced over the processes; w = w - sum_i h_i v_i
      CUBLAS_CHECK_OUTER( cublasZgemv( og.h, CUBLAS_OP_C, (int)vl, j+1, &one, og.V, (int)vl, w, 1, &zero, og.hd, 1 ) );
      CUDA_CHECK_OUTER( cudaMemcpy( hloc.data(), og.hd, (j+1)*sizeof(cu_cmplx_double), cudaMemcpyDeviceToHost ) );
      if ( g.num_processes > 1 )
        MPI_Allreduce( hloc.data(), hglob.data(), j+1, MPI_COMPLEX_double, MPI_SUM, g.comm_cart );
      else
        hglob = hloc;
      for ( int i=0; i<=j; i++ ) Hc(j,i) = cd_outer( hglob[i].x, hglob[i].y );
      CUDA_CHECK_OUTER( cudaMemcpy( og.hd, hglob.data(), (j+1)*sizeof(cu_cmplx_double), cudaMemcpyHostToDevice ) );
      CUBLAS_CHECK_OUTER( cublasZgemv( og.h, CUBLAS_OP_N, (int)vl, j+1, &mone, og.V, (int)vl, og.hd, 1, &one, w, 1 ) );

      // h_{j+1} = ||w||, v_{j+1} = w / h_{j+1}
      double hn = outer_norm( w, vl );
      Hc(j,j+1) = hn;
      if ( hn > 1e-15 ) { double s = 1.0/hn; CUBLAS_CHECK_OUTER( cublasZdscal( og.h, (int)vl, &s, w, 1 ) ); }

      if ( std::abs( Hc(j,j+1) ) > p->tol/10 ) {
        // Givens rotations (qr_update_double)
        for ( int i=0; i<j; i++ ) {
          cd_outer beta = (-sn[i])*Hc(j,i) + cs[i]*Hc(j,i+1);
          Hc(j,i) = std::conj(cs[i])*Hc(j,i) + std::conj(sn[i])*Hc(j,i+1);
          Hc(j,i+1) = beta;
        }
        double beta = sqrt( std::norm( Hc(j,j) ) + std::norm( Hc(j,j+1) ) );
        sn[j] = Hc(j,j+1)/beta; cs[j] = Hc(j,j)/beta;
        gam[j+1] = (-sn[j])*gam[j]; gam[j] = std::conj(cs[j])*gam[j];
        Hc(j,j) = beta; Hc(j,j+1) = 0;
        gamma_jp1 = std::abs( gam[j+1] );
#if defined(TRACK_RES) && !defined(WILSON_BENCHMARK)
        if ( p->print && g.print > 0 )
          printf0("| approx. rel. res. after  %-6d iterations: %e |\n", iter, gamma_jp1/norm_r0 );
#endif
        if ( gamma_jp1/norm_r0 < p->tol || gamma_jp1/norm_r0 > 1E+5 ) {
          og.finish = 1;
          if ( gamma_jp1/norm_r0 > 1E+5 ) printf0("Divergence of fgmres_double, iter = %d, level=%d\n", iter, l->level );
        }
      } else {
        printf0("depth: %d, iter: %d, p->H(%d,%d) = %+lf+%lfi\n", l->depth, iter, j+1, j,
                Hc(j,j+1).real(), Hc(j,j+1).imag() );
        og.finish = 1;
        og.stop_restart = 1;
      }
      END_LOCKED_MASTER(threading)
      SYNC_MASTER_TO_ALL(threading)
    }

    // x = x + Z y (x = Z y after a zero initial guess): compute_solution_double
    START_LOCKED_MASTER(threading)
    for ( int i=j; i>=0; i-- ) {
      y[i] = gam[i];
      for ( int k=i+1; k<=j; k++ ) y[i] -= Hc(k,i)*y[k];
      y[i] /= Hc(i,i);
    }
    for ( int i=0; i<=j; i++ ) hglob[i] = make_cuDoubleComplex( y[i].real(), y[i].imag() );
    CUDA_CHECK_OUTER( cudaMemcpy( og.hd, hglob.data(), (j+1)*sizeof(cu_cmplx_double), cudaMemcpyHostToDevice ) );
    int add = ( res == _NO_RES ) ? ol : 1;
    CUBLAS_CHECK_OUTER( cublasZgemv( og.h, CUBLAS_OP_N, (int)vl, j+1, &one, og.Z, (int)vl, og.hd, 1,
                                     add ? &one : &zero, og.x, 1 ) );
    CUDA_CHECK_OUTER( cudaDeviceSynchronize() );
    END_LOCKED_MASTER(threading)
    SYNC_MASTER_TO_ALL(threading)
  }

  START_LOCKED_MASTER(threading)
  t1 = MPI_Wtime(); g.total_time = t1-t0; g.iter_count = iter; g.norm_res = gamma_jp1/norm_r0;
  double beta = gamma_jp1;
#ifdef FGMRES_RESTEST
  // exact residual, on the GPU
  outer_apply_D( og.w, og.x, p->op, l, threading );
  CUBLAS_CHECK_OUTER( cublasZdscal( og.h, (int)vl, &minus1, og.w, 1 ) );
  CUBLAS_CHECK_OUTER( cublasZaxpy( og.h, (int)vl, &one, og.b, 1, og.w, 1 ) );
  beta = outer_norm( og.w, vl );
#endif
  // the only device-to-host copy of a fine vector in the solve
  CUDA_CHECK_OUTER( cudaMemcpy( p->x, og.x, vl*sizeof(cu_cmplx_double), cudaMemcpyDeviceToHost ) );

  if ( p->print ) {
    g.norm_res = beta/norm_r0;
#if defined(TRACK_RES) && !defined(WILSON_BENCHMARK)
    if ( g.print > 0 ) printf0("+----------------------------------------------------------+\n\n");
#endif
    printf0("+----------------------------------------------------------+\n");
    printf0("|       FGMRES iterations: %-6d coarse average: %-6.2lf   |\n", iter,
            ((double)g.coarse_iter_count)/((double)iter) );
    printf0("| exact relative residual: ||r||/||b|| = %e      |\n", beta/norm_r0 );
    printf0("| elapsed wall clock time: %-8.4lf seconds                |\n", t1-t0 );
    if ( g.coarse_time > 0 )
      printf0("|        coarse grid time: %-8.4lf seconds (%04.1lf%%)        |\n",
              g.coarse_time, 100*(g.coarse_time/(t1-t0)) );
    printf0("|        coarsest grid time: %-8.4lf seconds (%04.1lf%%)        |\n",
              g.coarsest_time, 100*(g.coarsest_time/(t1-t0)) );
    printf0("| coarsest grid matmul time: %-8.4lf seconds (%04.1lf%%)        |\n",
              g.matmul_time, 100*(g.matmul_time/(t1-t0)) );
    printf0("|  consumed core minutes*: %-8.2le (solve only)           |\n", ((t1-t0)*g.num_processes*MAX(1,threading->n_core))/60.0 );
    printf0("|    max used mem/MPIproc: %-8.2le GB                     |\n", g.max_storage/1024.0 );
    printf0("|        max used mem/GPU: %-8.2le GB                     |\n", g.max_gpu_storage/1024.0 );
    printf0("|  outer FGMRES on the GPU: %-3s                            |\n", "yes" );
    printf0("+----------------------------------------------------------+\n");
    printf0("*: only correct if #MPIprocs*#threads == #CPUs\n\n");
  }
  if ( p->timing || p->print ) prof_print( l );
  END_LOCKED_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)

  return iter;
}

#endif
