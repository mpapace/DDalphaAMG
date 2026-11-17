#ifndef PROFILING_OLD_PRECISION_H
#define PROFILING_OLD_PRECISION_H

#include "global_enums.h"

// -DPROF_NVTX (with PROFILING and without NVTX_DISABLE): every PROF_PRECISION_START/STOP pair
// is also an NVTX range, named after the timer and the depth, so that the CPU time of the timers
// is visible in Nsight Systems. Only the master thread opens ranges in the threaded variants.
#if defined(PROFILING) && defined(PROF_NVTX) && !defined(NVTX_DISABLE)
#include <nvtx3/nvToolsExt.h>
#define PROF_PRECISION_NVTX_PUSH(TYPE) nvtxRangePushA(l->prof_PRECISION.nvtx_name[TYPE])
#define PROF_PRECISION_NVTX_POP()      nvtxRangePop()
#else
#define PROF_PRECISION_NVTX_PUSH(TYPE)
#define PROF_PRECISION_NVTX_POP()
#endif

typedef struct
{
    double time[_NUM_PROF];
    double flop[_NUM_PROF];
    double count[_NUM_PROF];
    char name[_NUM_PROF][50];
    char nvtx_name[_NUM_PROF][64];  // "d<depth> <name>", label of the NVTX range (PROF_NVTX)
} profiling_PRECISION_struct;

#ifdef PROFILING
#define PROF_PRECISION_START_UNTHREADED(TYPE)        \
    do                                               \
    {                                                \
        PROF_PRECISION_NVTX_PUSH(TYPE);                        \
        l->prof_PRECISION.time[TYPE] -= MPI_Wtime(); \
    } while (0)
#define PROF_PRECISION_START_THREADED(TYPE, threading)   \
    do                                                   \
    {                                                    \
        if (threading->core + threading->thread == 0)    \
        {                                                \
            PROF_PRECISION_NVTX_PUSH(TYPE);                        \
            l->prof_PRECISION.time[TYPE] -= MPI_Wtime(); \
        }                                                \
    } while (0)
#else
#define PROF_PRECISION_START_UNTHREADED(TYPE)
#define PROF_PRECISION_START_THREADED(TYPE, threading)
#endif

#ifdef PROFILING
#define PROF_PRECISION_STOP_UNTHREADED(TYPE, COUNT)  \
    do                                               \
    {                                                \
        l->prof_PRECISION.time[TYPE] += MPI_Wtime(); \
        l->prof_PRECISION.count[TYPE] += COUNT;      \
        PROF_PRECISION_NVTX_POP();                             \
    } while (0)
#define PROF_PRECISION_STOP_THREADED(TYPE, COUNT, threading) \
    do                                                       \
    {                                                        \
        if (threading->core + threading->thread == 0)        \
        {                                                    \
            l->prof_PRECISION.time[TYPE] += MPI_Wtime();     \
            l->prof_PRECISION.count[TYPE] += COUNT;          \
            PROF_PRECISION_NVTX_POP();                                 \
        }                                                    \
    } while (0)
#else
#define PROF_PRECISION_STOP_UNTHREADED(TYPE, COUNT)
#define PROF_PRECISION_STOP_THREADED(TYPE, COUNT, threading)
#endif

#define GET_MACRO2(_1, _2, NAME, ...) NAME
#define GET_MACRO3(_1, _2, _3, NAME, ...) NAME
#define PROF_PRECISION_START(...) GET_MACRO2(__VA_ARGS__, PROF_PRECISION_START_THREADED, PROF_PRECISION_START_UNTHREADED, padding)(__VA_ARGS__)
#define PROF_PRECISION_STOP(...) GET_MACRO3(__VA_ARGS__, PROF_PRECISION_STOP_THREADED, PROF_PRECISION_STOP_UNTHREADED, padding)(__VA_ARGS__)

#endif // PROFILING_OLD_PRECISION_H
