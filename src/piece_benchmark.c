/*
 * Benchmark of the pieces ported to the GPU against their CPU code (build with -DCUDA_PIECE_BENCHMARK).
 *
 * Called once after the solve, by all threads (the work is done by the master thread, with the CPU code
 * run without OpenMP threads: the runs of the benchmark use 1 thread per rank). The state of the solver
 * is overwritten, so nothing can be solved after it. Inputs are random vectors, the operators are the
 * ones of the run. Every piece is run PIECE_BENCH_REPS times after a barrier and timed with MPI_Wtime;
 * rank 0 prints the medians, in ms, one line per piece:
 *
 *   piece bench: depth=<d> piece=<name> cpu_ms=<CPU code> gpu_ms=<GPU, vectors on the GPU> gpu_host_ms=<GPU with
 *                the copies of the vectors from/to the host, -1 if the piece has no such version>
 *
 * Pieces (single precision, method 4 with odd-even, mixed precision), coarse levels with their operators on
 * the GPU:
 *   depth>0  1A_schur        Schur complement of the odd-even operator
 *            1B_coarse_op    coarse operator (l->s_PRECISION.op)
 *            1C_first_last   first and last step of the odd-even smoother (diag_oo_inv, n_hopping)
 *            1D_gmres        GMRES of the odd-even smoother (CPU: fgmres with the CPU Schur complement)
 *            1_smoother      whole odd-even smoother: CPU 1C + 1D in one call sequence, GPU
 *                            cuda_coarse_solve_odd_even (with the copies of b and x)
 * Restriction and interpolation: -DCUDA_INTERP_BENCHMARK (lines "interp bench: ...").
 */

#include "main.h"
#include "piece_benchmark.h"

#if defined( CUDA_OPT ) && defined( CUDA_PIECE_BENCHMARK )

#define PIECE_BENCH_REPS 20

void cuda_coarse_oddeven_float_bench( gmres_float_struct *p, vector_float in_h, vector_float out_h,
                                      level_struct *l, int reps, double *t );

static int piece_bench_cmp( const void *a, const void *b ) {
  double x = *(const double*)a, y = *(const double*)b;
  return (x > y) - (x < y);
}

static double piece_bench_median( double *t, int n ) {
  qsort( t, n, sizeof(double), piece_bench_cmp );
  return t[n/2];
}

static void piece_bench_print( int depth, const char *piece, double cpu, double gpu, double gpu_host ) {
  printf0( "piece bench: depth=%d piece=%s cpu_ms=%.4f gpu_ms=%.4f gpu_host_ms=%.4f\n",
           depth, piece, cpu, gpu, gpu_host );
}

// times (ms) of the CPU code of the statement, reps times after a barrier; median in dst
#define PIECE_BENCH_CPU( dst, ... )                                    \
  do {                                                                  \
    double t_[PIECE_BENCH_REPS];                                        \
    for ( int r_=0; r_<PIECE_BENCH_REPS; r_++ ) {                       \
      MPI_Barrier( g.comm_cart );                                       \
      double t0_ = MPI_Wtime();                                         \
      __VA_ARGS__;                                                      \
      t_[r_] = 1e3*( MPI_Wtime() - t0_ );                               \
    }                                                                   \
    dst = piece_bench_median( t_, PIECE_BENCH_REPS );                   \
  } while (0)


// a coarse level (depth > 0) with its operators on the GPU
static void piece_bench_coarse( level_struct *l ) {

  struct Thread *th = no_threading;
  gmres_float_struct *sp = &(l->sp_float);
  operator_float_struct *op = &(l->oe_op_float), *sop = &(l->s_float.op);
  int n = l->inner_vector_size, nv = l->vector_size;
  vector_float in = NULL, out = NULL;
  double cpu[5], t_gpu[7];

  MALLOC( in, complex_float, nv );
  MALLOC( out, complex_float, nv );
  vector_float_define_random( in, 0, n, l );

  int num_restart = sp->num_restart, igz = sp->initial_guess_zero;
  void (*eval_operator)() = sp->eval_operator;
  sp->num_restart = l->post_smooth_iter;
  sp->eval_operator = coarse_apply_schur_complement_float_cpu;

  PIECE_BENCH_CPU( cpu[0], coarse_apply_schur_complement_float_cpu( out, in, op, l, th ) );
  PIECE_BENCH_CPU( cpu[1], apply_coarse_operator_float_cpu( out, in, sop, l, th ) );
  // first and last step of coarse_solve_odd_even_float (CPU code)
  PIECE_BENCH_CPU( cpu[2], vector_float_copy( sp->b, in, 0, n, l );
                           coarse_diag_oo_inv_float( sp->x, sp->b, op, l, th );
                           coarse_n_hopping_term_float( sp->b, sp->x, op, _EVEN_SITES, l, th );
                           coarse_n_hopping_term_float( sp->b, sp->x, op, _ODD_SITES, l, th );
                           coarse_diag_oo_inv_float( sp->x, sp->b, op, l, th ) );
  vector_float_copy( sp->b, in, 0, n, l );
  PIECE_BENCH_CPU( cpu[3], sp->initial_guess_zero = _NO_RES; fgmres_float( sp, l, th ) );
  PIECE_BENCH_CPU( cpu[4], vector_float_copy( sp->b, in, 0, n, l );
                           sp->initial_guess_zero = _NO_RES;
                           coarse_diag_oo_inv_float( sp->x, sp->b, op, l, th );
                           coarse_n_hopping_term_float( sp->b, sp->x, op, _EVEN_SITES, l, th );
                           fgmres_float( sp, l, th );
                           coarse_n_hopping_term_float( sp->b, sp->x, op, _ODD_SITES, l, th );
                           coarse_diag_oo_inv_float( sp->x, sp->b, op, l, th ) );

  sp->eval_operator = eval_operator;
  cuda_coarse_oddeven_float_bench( sp, in, out, l, PIECE_BENCH_REPS, t_gpu );
  sp->num_restart = num_restart;
  sp->initial_guess_zero = igz;

  piece_bench_print( l->depth, "1A_schur",      cpu[0], t_gpu[0], t_gpu[4] );
  piece_bench_print( l->depth, "1B_coarse_op",  cpu[1], t_gpu[1], t_gpu[5] );
  piece_bench_print( l->depth, "1C_first_last", cpu[2], t_gpu[2], -1 );
  piece_bench_print( l->depth, "1D_gmres",      cpu[3], t_gpu[3], -1 );
  piece_bench_print( l->depth, "1_smoother",    cpu[4], t_gpu[2] + t_gpu[3], t_gpu[6] );

  FREE( in, complex_float, nv );
  FREE( out, complex_float, nv );
}


void piece_benchmark( level_struct *l, struct Thread *threading ) {

  START_LOCKED_MASTER(threading)
  if ( g.mixed_precision != 1 || g.method != 4 || !g.odd_even || !g.interpolation ) {
    printf0("piece bench: needs mixed precision, method 4, odd-even and interpolation, skipped\n");
  } else {
    printf0("piece bench: %d processes, %d repetitions per piece, medians in ms\n", g.num_processes, PIECE_BENCH_REPS );
    for ( level_struct *lc = l->next_level; lc != NULL && lc->level > 0; lc = lc->next_level ) {
      // the benchmark needs all the processes (barriers on g.comm_cart) and the operators on the GPU
      int skip = lc->idle || !cuda_coarse_oddeven_on_gpu_float( lc ) ||
                 !cuda_coarse_operator_on_gpu_float( &(lc->s_float.op), lc ), skip_any = skip;
      MPI_Allreduce( &skip, &skip_any, 1, MPI_INT, MPI_MAX, g.comm_cart );
      if ( skip_any )
        printf0("piece bench: depth=%d skipped (idle processes or operators not on the GPU)\n", lc->depth );
      else
        piece_bench_coarse( lc );
    }
  }
  END_LOCKED_MASTER(threading)
}

#endif
