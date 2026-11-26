#include "cuda_complex.h"
#include "cuda_complex_cxx.h"
#include "cuda_complex_operators.h"
#include "cuda_complex_operators_PRECISION.h"
#include "cuda_componentwise.h"
#include "cuda_dirac_kernels_componentwise_PRECISION.h"
#include "cuda_half.h"
#include "global_enums.h"

// The clifford header uses a C compiler extension version of I that is not compatible with CUDA.
// CU_OVERWRITE_I replaces that.
#define CU_OVERWRITE_I
#include "clifford.h"
#undef CU_OVERWRITE_I

__global__ void cuda_site_clover_componentwise_PRECISION(cuda_vector_PRECISION eta,
                                                         cu_cmplx_PRECISION const* phi,
                                                         cu_cmplx_PRECISION const* clover,
                                                         size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  auto caEta = ComponentAccess(eta + idx, num_sites);
  auto caPhi = ComponentAccess(phi + idx, num_sites);
  // clover += idx;
  auto caClover = ComponentAccess(clover + idx, num_sites);

  // diagonal
  caEta[0] = caClover[0] * caPhi[0];
  caEta[1] = caClover[1] * caPhi[1];
  caEta[2] = caClover[2] * caPhi[2];
  caEta[3] = caClover[3] * caPhi[3];
  caEta[4] = caClover[4] * caPhi[4];
  caEta[5] = caClover[5] * caPhi[5];
  caEta[6] = caClover[6] * caPhi[6];
  caEta[7] = caClover[7] * caPhi[7];
  caEta[8] = caClover[8] * caPhi[8];
  caEta[9] = caClover[9] * caPhi[9];
  caEta[10] = caClover[10] * caPhi[10];
  caEta[11] = caClover[11] * caPhi[11];
  // spin 0 and 1, row major
  caEta[0] += caClover[12] * caPhi[1];
  caEta[0] += caClover[13] * caPhi[2];
  caEta[0] += caClover[14] * caPhi[3];
  caEta[0] += caClover[15] * caPhi[4];
  caEta[0] += caClover[16] * caPhi[5];
  caEta[1] += caClover[17] * caPhi[2];
  caEta[1] += caClover[18] * caPhi[3];
  caEta[1] += caClover[19] * caPhi[4];
  caEta[1] += caClover[20] * caPhi[5];
  caEta[2] += caClover[21] * caPhi[3];
  caEta[2] += caClover[22] * caPhi[4];
  caEta[2] += caClover[23] * caPhi[5];
  caEta[3] += caClover[24] * caPhi[4];
  caEta[3] += caClover[25] * caPhi[5];
  caEta[4] += caClover[26] * caPhi[5];
  caEta[1] += cu_conj_PRECISION(caClover[12]) * caPhi[0];
  caEta[2] += cu_conj_PRECISION(caClover[13]) * caPhi[0];
  caEta[3] += cu_conj_PRECISION(caClover[14]) * caPhi[0];
  caEta[4] += cu_conj_PRECISION(caClover[15]) * caPhi[0];
  caEta[5] += cu_conj_PRECISION(caClover[16]) * caPhi[0];
  caEta[2] += cu_conj_PRECISION(caClover[17]) * caPhi[1];
  caEta[3] += cu_conj_PRECISION(caClover[18]) * caPhi[1];
  caEta[4] += cu_conj_PRECISION(caClover[19]) * caPhi[1];
  caEta[5] += cu_conj_PRECISION(caClover[20]) * caPhi[1];
  caEta[3] += cu_conj_PRECISION(caClover[21]) * caPhi[2];
  caEta[4] += cu_conj_PRECISION(caClover[22]) * caPhi[2];
  caEta[5] += cu_conj_PRECISION(caClover[23]) * caPhi[2];
  caEta[4] += cu_conj_PRECISION(caClover[24]) * caPhi[3];
  caEta[5] += cu_conj_PRECISION(caClover[25]) * caPhi[3];
  caEta[5] += cu_conj_PRECISION(caClover[26]) * caPhi[4];
  // spin 2 and 3, row major
  caEta[6] += caClover[27] * caPhi[7];
  caEta[6] += caClover[28] * caPhi[8];
  caEta[6] += caClover[29] * caPhi[9];
  caEta[6] += caClover[30] * caPhi[10];
  caEta[6] += caClover[31] * caPhi[11];
  caEta[7] += caClover[32] * caPhi[8];
  caEta[7] += caClover[33] * caPhi[9];
  caEta[7] += caClover[34] * caPhi[10];
  caEta[7] += caClover[35] * caPhi[11];
  caEta[8] += caClover[36] * caPhi[9];
  caEta[8] += caClover[37] * caPhi[10];
  caEta[8] += caClover[38] * caPhi[11];
  caEta[9] += caClover[39] * caPhi[10];
  caEta[9] += caClover[40] * caPhi[11];
  caEta[10] += caClover[41] * caPhi[11];
  caEta[7] += cu_conj_PRECISION(caClover[27]) * caPhi[6];
  caEta[8] += cu_conj_PRECISION(caClover[28]) * caPhi[6];
  caEta[9] += cu_conj_PRECISION(caClover[29]) * caPhi[6];
  caEta[10] += cu_conj_PRECISION(caClover[30]) * caPhi[6];
  caEta[11] += cu_conj_PRECISION(caClover[31]) * caPhi[6];
  caEta[8] += cu_conj_PRECISION(caClover[32]) * caPhi[7];
  caEta[9] += cu_conj_PRECISION(caClover[33]) * caPhi[7];
  caEta[10] += cu_conj_PRECISION(caClover[34]) * caPhi[7];
  caEta[11] += cu_conj_PRECISION(caClover[35]) * caPhi[7];
  caEta[9] += cu_conj_PRECISION(caClover[36]) * caPhi[8];
  caEta[10] += cu_conj_PRECISION(caClover[37]) * caPhi[8];
  caEta[11] += cu_conj_PRECISION(caClover[38]) * caPhi[8];
  caEta[10] += cu_conj_PRECISION(caClover[39]) * caPhi[9];
  caEta[11] += cu_conj_PRECISION(caClover[40]) * caPhi[9];
  caEta[11] += cu_conj_PRECISION(caClover[41]) * caPhi[10];
}

__global__ void cuda_site_diag_ee_componentwise_PRECISION(cuda_vector_PRECISION eta,
                                                          cu_cmplx_PRECISION const* phi,
                                                          cu_cmplx_PRECISION const* clover,
                                                          size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  auto caEta = ComponentAccess(eta + idx, num_sites);
  auto caPhi = ComponentAccess(phi + idx, num_sites);
  // clover += idx;
  auto caClover = ComponentAccess(clover + idx, num_sites);

  int i, j, k=0, n;
  cu_cmplx_PRECISION z[6];

  for ( n=0; n<2; n++ ) {
    // z = L^H x
    for ( j=0; j<6; j++ ) { // columns
      for ( i=0; i<j; i++ ) { // rows
        //z[i] += conj_PRECISION(*L)*x[j]; L++;
        z[i] += cu_conj_PRECISION(caClover[k])*caPhi[n*6+j]; k++;
      }

      //z[j] = conj_PRECISION(*L)*x[j]; L++;
      z[j] = cu_conj_PRECISION(caClover[k])*caPhi[n*6+j]; k++;
    }
    //L-=21;
    k -= 21;
    // y = L*z;
    for ( i=0; i<6; i++ ) { // rows
      //y[i] = *L * z[0]; L++;
      caEta[n*6+i] = caClover[k]*z[0]; k++;
      for ( j=1; j<=i; j++ ) { // columns
        //y[i] += *L * z[j]; L++;
        caEta[n*6+i] += caClover[k]*z[j]; k++;
      }
    }
    //x+=6;
    //y+=6;
  }
}

__global__ void cuda_site_diag_oo_inv_componentwise_PRECISION(cuda_vector_PRECISION eta,
                                                              cu_cmplx_PRECISION const* phi,
                                                              cu_cmplx_PRECISION const* clover,
                                                              size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  auto caEta = ComponentAccess(eta + idx, num_sites);
  auto caPhi = ComponentAccess(phi + idx, num_sites);
  // clover += idx;
  auto caClover = ComponentAccess(clover + idx, num_sites);

  int i, j, k=0, n;

  for ( n=0; n<2; n++ ) {
    // forward substitution with L
    for ( i=0; i<6; i++ ) {
      //x[i] = b[i];
      caEta[n*6+i] = caPhi[n*6+i];
      for ( j=0; j<i; j++ ) {
        //x[i] = x[i] - *L * x[j]; L++;
        caEta[n*6+i] = caEta[n*6+i] - caClover[k]*caEta[n*6+j]; k++;
      }
      //x[i] = x[i] / *L; L++;
      caEta[n*6+i] = cu_cdiv_PRECISION(caEta[n*6+i],caClover[k]); k++;
    }
    //L -= 21;
    k -= 21;
    // backward substitution with L^H
    for ( i=5; i>=0; i-- ) {
      for ( j=i+1; j<6; j++ ) {
        //x[i] = x[i] - conj_PRECISION(L[(j*(j+1))/2 + i]) * x[j];
        caEta[n*6+i] = caEta[n*6+i] - cu_conj_PRECISION(caClover[k+(j*(j+1))/2+i])*caEta[n*6+j];
      }
      //x[i] = x[i] / conj_PRECISION(L[(i*(i+1))/2 + i]);
      caEta[n*6+i] = cu_cdiv_PRECISION(caEta[n*6+i],cu_conj_PRECISION(caClover[k+(i*(i+1))/2+i]));
    }
    //x+=6;
    //b+=6;
    //L+=21;
    k += 21;
  }
}

// ---------------------------------------------------------------------------------------------
// Fused hopping term
// ---------------------------------------------------------------------------------------------

// 6 consecutive complex numbers (one chunk of a prp/prn buffer): 16-byte loads and stores.
// The chunks start at multiples of 6 elements of a cudaMalloc'ed buffer, so they are 16-byte aligned.
__device__ inline void hopping_load6_PRECISION(cu_cmplx_PRECISION* v, cu_cmplx_PRECISION const* p) {
  if constexpr (sizeof(cu_cmplx_PRECISION) == 8) {
    float4 const* q = reinterpret_cast<float4 const*>(p);
#pragma unroll
    for (int k = 0; k < 3; k++) {
      const float4 a = q[k];
      v[2 * k].x = a.x; v[2 * k].y = a.y; v[2 * k + 1].x = a.z; v[2 * k + 1].y = a.w;
    }
  } else {
#pragma unroll
    for (int k = 0; k < 6; k++) v[k] = p[k];
  }
}

__device__ inline void hopping_store6_PRECISION(cu_cmplx_PRECISION* p, cu_cmplx_PRECISION const* v) {
  if constexpr (sizeof(cu_cmplx_PRECISION) == 8) {
    float4* q = reinterpret_cast<float4*>(p);
#pragma unroll
    for (int k = 0; k < 3; k++) q[k] = make_float4(v[2 * k].x, v[2 * k].y, v[2 * k + 1].x, v[2 * k + 1].y);
  } else {
#pragma unroll
    for (int k = 0; k < 6; k++) p[k] = v[k];
  }
}

// the 9 entries of U_mu at site idx, from the float (HALF = false) or half-precision links;
// pointers passed by value (taking the address of the kernel parameter struct would copy it to
// local memory)
template <bool HALF>
__device__ inline void hopping_load_links_PRECISION(cu_cmplx_PRECISION* U, cu_cmplx_PRECISION const* D,
                                                    __half2 const* Dh, float Dh_scale, size_t idx,
                                                    size_t num_sites) {
  if constexpr (HALF) {
    auto caD = make_op_access<cu_cmplx_PRECISION>(Dh + idx, num_sites, Dh_scale);
#pragma unroll
    for (int k = 0; k < 9; k++) U[k] = caD[k];
  } else {
    auto caD = ComponentAccess(D + idx, num_sites);
#pragma unroll
    for (int k = 0; k < 9; k++) U[k] = caD[k];
  }
}

// projection side for one direction (DIR = T, Z, Y or X, mu its index)
#define HOPPING_PROJECT_DIR_PRECISION( DIR, mu ) do {                                            \
    cu_cmplx_PRECISION pm[6], pp[6], r[6], U[9];                                                \
    _Pragma("unroll")                                                                            \
    for (int c = 0; c < 3; c++) {                                                                \
      pm[c]     = x[c]     - GAMMA_##DIR##_SPIN0_VAL * x[3 * GAMMA_##DIR##_SPIN0_CO + c];         \
      pm[3 + c] = x[3 + c] - GAMMA_##DIR##_SPIN1_VAL * x[3 * GAMMA_##DIR##_SPIN1_CO + c];         \
      pp[c]     = x[c]     + GAMMA_##DIR##_SPIN0_VAL * x[3 * GAMMA_##DIR##_SPIN0_CO + c];         \
      pp[3 + c] = x[3 + c] + GAMMA_##DIR##_SPIN1_VAL * x[3 * GAMMA_##DIR##_SPIN1_CO + c];         \
    }                                                                                            \
    hopping_store6_PRECISION(args.prn[mu] + 6 * (start + idx), pm);                              \
    hopping_load_links_PRECISION<HALF>(U, args.D[mu], args.Dh[mu], args.Dh_scale[mu], idx, num_sites);\
    /* r = U^H pp, for both spin halves */                                                       \
    _Pragma("unroll")                                                                            \
    for (int h = 0; h < 2; h++) {                                                                \
      _Pragma("unroll")                                                                          \
      for (int i = 0; i < 3; i++) {                                                              \
        r[3 * h + i] = cu_conj_PRECISION(U[i]) * pp[3 * h];                                      \
        r[3 * h + i] += cu_conj_PRECISION(U[3 + i]) * pp[3 * h + 1];                             \
        r[3 * h + i] += cu_conj_PRECISION(U[6 + i]) * pp[3 * h + 2];                             \
      }                                                                                          \
    }                                                                                            \
    hopping_store6_PRECISION(args.prp[mu] + 6 * (size_t) neighbors[4 * idx + mu], r);           \
  } while (0)

template <bool HALF>
__device__ __forceinline__ void cuda_hopping_project_componentwise_body_PRECISION(cuda_hopping_args_PRECISION args,
                                                             cu_cmplx_PRECISION const* phi,
                                                             int const* neighbors, size_t start,
                                                             size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  auto caPhi = ComponentAccess(phi + idx, num_sites);
  cu_cmplx_PRECISION x[12];
#pragma unroll
  for (int i = 0; i < 12; i++) x[i] = caPhi[i];

  HOPPING_PROJECT_DIR_PRECISION(T, 0);
  HOPPING_PROJECT_DIR_PRECISION(Z, 1);
  HOPPING_PROJECT_DIR_PRECISION(Y, 2);
  HOPPING_PROJECT_DIR_PRECISION(X, 3);
}

// gather side for one direction (DIR = T, Z, Y or X, mu its index)
#define HOPPING_GATHER_DIR_PRECISION( DIR, mu ) do {                                             \
    cu_cmplx_PRECISION g[6], u[6], U[9];                                                        \
    /* forward hop: u = U_mu (1 - gamma_mu) phi(x + mu), lifted as in pbp_su3 */                 \
    hopping_load6_PRECISION(g, args.prn[mu] + 6 * (size_t) neighbors[4 * idx + mu]);            \
    hopping_load_links_PRECISION<HALF>(U, args.D[mu], args.Dh[mu], args.Dh_scale[mu], idx, num_sites);\
    _Pragma("unroll")                                                                            \
    for (int h = 0; h < 2; h++) {                                                                \
      _Pragma("unroll")                                                                          \
      for (int i = 0; i < 3; i++) {                                                              \
        u[3 * h + i] = U[3 * i] * g[3 * h];                                                      \
        u[3 * h + i] += U[3 * i + 1] * g[3 * h + 1];                                             \
        u[3 * h + i] += U[3 * i + 2] * g[3 * h + 2];                                             \
      }                                                                                          \
    }                                                                                            \
    _Pragma("unroll")                                                                            \
    for (int c = 0; c < 3; c++) {                                                                \
      e[c] -= u[c];                                                                              \
      e[3 + c] -= u[3 + c];                                                                      \
      e[6 + c] += GAMMA_##DIR##_SPIN2_VAL * u[3 * GAMMA_##DIR##_SPIN2_CO + c];                   \
      e[9 + c] += GAMMA_##DIR##_SPIN3_VAL * u[3 * GAMMA_##DIR##_SPIN3_CO + c];                   \
    }                                                                                            \
    /* backward hop: U_mu^H (1 + gamma_mu) phi(x - mu), already in prp at this site (pbn_su3) */ \
    hopping_load6_PRECISION(g, args.prp[mu] + 6 * (start + idx));                                \
    _Pragma("unroll")                                                                            \
    for (int c = 0; c < 3; c++) {                                                                \
      e[c] -= g[c];                                                                              \
      e[3 + c] -= g[3 + c];                                                                      \
      e[6 + c] -= GAMMA_##DIR##_SPIN2_VAL * g[3 * GAMMA_##DIR##_SPIN2_CO + c];                   \
      e[9 + c] -= GAMMA_##DIR##_SPIN3_VAL * g[3 * GAMMA_##DIR##_SPIN3_CO + c];                   \
    }                                                                                            \
  } while (0)

template <bool HALF>
__device__ __forceinline__ void cuda_hopping_gather_componentwise_body_PRECISION(cu_cmplx_PRECISION* eta,
                                                            cuda_hopping_args_PRECISION args,
                                                            int const* neighbors, size_t start,
                                                            size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  auto caEta = ComponentAccess(eta + idx, num_sites);
  cu_cmplx_PRECISION e[12];
#pragma unroll
  for (int i = 0; i < 12; i++) e[i] = caEta[i];

  HOPPING_GATHER_DIR_PRECISION(T, 0);
  HOPPING_GATHER_DIR_PRECISION(Z, 1);
  HOPPING_GATHER_DIR_PRECISION(Y, 2);
  HOPPING_GATHER_DIR_PRECISION(X, 3);

#pragma unroll
  for (int i = 0; i < 12; i++) caEta[i] = e[i];
}

__global__ void cuda_hopping_project_componentwise_PRECISION(cuda_hopping_args_PRECISION args,
                                                             cu_cmplx_PRECISION const* phi,
                                                             int const* neighbors, size_t start,
                                                             size_t num_sites) {
  cuda_hopping_project_componentwise_body_PRECISION<false>(args, phi, neighbors, start, num_sites);
}
__global__ void cuda_hopping_project_componentwise_half_PRECISION(cuda_hopping_args_PRECISION args,
                                                             cu_cmplx_PRECISION const* phi,
                                                             int const* neighbors, size_t start,
                                                             size_t num_sites) {
  cuda_hopping_project_componentwise_body_PRECISION<true>(args, phi, neighbors, start, num_sites);
}

__global__ void cuda_hopping_gather_componentwise_PRECISION(cu_cmplx_PRECISION* eta,
                                                            cuda_hopping_args_PRECISION args,
                                                            int const* neighbors, size_t start,
                                                            size_t num_sites) {
  cuda_hopping_gather_componentwise_body_PRECISION<false>(eta, args, neighbors, start, num_sites);
}
__global__ void cuda_hopping_gather_componentwise_half_PRECISION(cu_cmplx_PRECISION* eta,
                                                            cuda_hopping_args_PRECISION args,
                                                            int const* neighbors, size_t start,
                                                            size_t num_sites) {
  cuda_hopping_gather_componentwise_body_PRECISION<true>(eta, args, neighbors, start, num_sites);
}
