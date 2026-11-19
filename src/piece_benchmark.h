#ifndef PIECE_BENCHMARK_HEADER
#define PIECE_BENCHMARK_HEADER

#if defined( CUDA_OPT ) && defined( CUDA_PIECE_BENCHMARK )
  // benchmark of the pieces ported to the GPU against their CPU code, after the solve (see piece_benchmark.c)
  void piece_benchmark( level_struct *l, struct Thread *threading );
#endif

#endif
