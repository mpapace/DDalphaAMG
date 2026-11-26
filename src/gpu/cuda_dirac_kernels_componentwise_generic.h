/**
 * \file cuda_dirac_kernels_componentwise_generic.h
 *
 * \brief CUDA kernels to perform calculations related to application of the Wilson-Dirac operator.
 */

#ifndef CUDA_DIRAC_KERNELS_COMPONENTWISE_PRECISION_H
#define CUDA_DIRAC_KERNELS_COMPONENTWISE_PRECISION_H

#include "cuda_vectors_PRECISION.h"
#include "global_enums.h"

// Better to have this known in advance, so we can predetermine shared
// memory requirements.
constexpr size_t diracCommonBlockSize = 128;

/**
 * \brief Apply the Clover term per lattice site.
 *
 * \param[out]  eta         Set to eta = C * phi.
 *                          I.e. is the result of applying the Clover term on phi.
 * \param[in]   phi         Quark field phi.
 * \param[in]   clover      The configuration of the Clover term.
 * \param[in]   num_sites   The number of lattice sites to process. A call which starts more threads
 *                          than needed for the number of lattice sites will have the remaining
 *                          threads idle/return.
 */
__global__ void cuda_site_clover_componentwise_PRECISION(cuda_vector_PRECISION eta,
                                                         cu_cmplx_PRECISION const* phi,
                                                         cu_cmplx_PRECISION const* clover,
                                                         size_t num_sites);

__global__ void cuda_site_diag_ee_componentwise_PRECISION(cuda_vector_PRECISION eta,
                                                          cu_cmplx_PRECISION const* phi,
                                                          cu_cmplx_PRECISION const* clover,
                                                          size_t num_sites);
__global__ void cuda_site_diag_oo_inv_componentwise_PRECISION(cuda_vector_PRECISION eta,
                                                              cu_cmplx_PRECISION const* phi,
                                                              cu_cmplx_PRECISION const* clover,
                                                              size_t num_sites);

/**
 * Fused hopping term (one kernel for the projection side, one for the gather side).
 *
 * Buffers (op->prn*_gpu, op->prp*_gpu): prn[mu] holds (1 - gamma_mu) phi at the site itself,
 * prp[mu] holds U_mu^H (1 + gamma_mu) phi scattered to the +mu neighbor; both in chunks of 6 per
 * site, indexed by the site index of the whole lattice (including the ghost region), which is the
 * layout the ghost exchange expects. Directions: 0 = T, 1 = Z, 2 = Y, 3 = X.
 */
typedef struct {
  cu_cmplx_PRECISION* prn[4];      // (1 - gamma_mu) phi, at the site
  cu_cmplx_PRECISION* prp[4];      // U_mu^H (1 + gamma_mu) phi, at the +mu neighbor
  cu_cmplx_PRECISION const* D[4];  // links, componentwise with stride num_sites, offset to the first site
} cuda_hopping_args_PRECISION;

/**
 * Projection side. Thread i handles site start+i of the source parity (or of the whole lattice):
 * reads phi once (componentwise, stride num_sites, phi offset to the first site) and writes
 * prn[mu][start+i] and prp[mu][neighbors[4*i+mu]] for the four directions.
 * neighbors must be offset to the first site (neighbor_table + 4*start).
 */
__global__ void cuda_hopping_project_componentwise_PRECISION(cuda_hopping_args_PRECISION args,
                                                             cu_cmplx_PRECISION const* phi,
                                                             int const* neighbors, size_t start,
                                                             size_t num_sites);

/**
 * Gather side. Thread i handles site start+i of the target parity (or of the whole lattice):
 * eta += sum_mu [ lift_+(U_mu prn[mu][neighbors[4*i+mu]]) + lift_-(prp[mu][start+i]) ],
 * with eta read and written once (componentwise, stride num_sites, offset to the first site).
 */
__global__ void cuda_hopping_gather_componentwise_PRECISION(cu_cmplx_PRECISION* eta,
                                                            cuda_hopping_args_PRECISION args,
                                                            int const* neighbors, size_t start,
                                                            size_t num_sites);

#endif  // CUDA_DIRAC_KERNELS_COMPONENTWISE_PRECISION_H
