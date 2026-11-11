#pragma once

#include "common.cuh"
#include "mmq.cuh"

#include <cstdint>

#define CUDA_QUANTIZE_BLOCK_SIZE     256
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

static_assert(MATRIX_ROW_PADDING %    CUDA_QUANTIZE_BLOCK_SIZE      == 0, "Risk of out-of-bounds access.");
static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

// x is value iqs of the q8_1 block y, the QK8_1 values of a block are held by consecutive threads of a warp
static __device__ __forceinline__ void quantize_q8_1_value(block_q8_1 * y, const int iqs, const float x) {
    const float amax = warp_reduce_max<QK8_1>(fabsf(x));
    const float sum  = warp_reduce_sum<QK8_1>(x);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(x / d);

    y->qs[iqs] = q;
    if (iqs == 0) {
        y->ds = make_half2(d, sum);
    }
}

// x is value iqs of the q8_1 block y and of the 16 bit block y16 (read by the MS_C mat-vec), y as with quantize_q8_1_value
static __device__ __forceinline__ void quantize_q8_1_y16_value(block_q8_1 * y, block_y16 * y16, const int iqs, const float x) {
    const float amax = warp_reduce_max<QK8_1>(fabsf(x));
    const float sum  = warp_reduce_sum<QK8_1>(x);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(x / d);

    y->qs[iqs] = q;
    if (iqs == 0) {
        y->ds = make_half2(d, sum);
    }

    const float d16 = amax / 32767.0f;
    y16->qs[iqs] = amax == 0.0f ? 0 : __float2int_rn(x / d16);
    if (iqs == 0) {
        y16->d = d16;
    }
}

static inline bool ggml_cuda_type_is_ms_c(const ggml_type type) {
    return type == GGML_TYPE_MS_C1 || type == GGML_TYPE_MS_C2 || type == GGML_TYPE_MS_C3 ||
           type == GGML_TYPE_MS_C4 || type == GGML_TYPE_MS_C5 || type == GGML_TYPE_MS_C6;
}

typedef void (*quantize_cuda_t)(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

// for MS_C src0 the block_y16 data follows the q8_1 blocks: ne0*ne1*ne2*ne3/QK8_1 blocks of each
void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_mmq_fp4_cuda(const float *   x,
                             const int32_t * ids,
                             void *          vy,
                             float *         scale,
                             ggml_type       type_src0,
                             bool            use_aligned_float8,
                             int64_t         ne00,
                             int64_t         s01,
                             int64_t         s02,
                             int64_t         s03,
                             int64_t         ne0,
                             int64_t         ne1,
                             int64_t         ne2,
                             int64_t         ne3,
                             cudaStream_t    stream);

// quantize each token once and scatter the block to its compact rows (via the inverse map)
void quantize_scatter_mmq_fp4_cuda(const float *   x,
                                   const int32_t * ids_src1_inv,
                                   void *          vy,
                                   float *         scale,
                                   ggml_type       type_src0,
                                   bool            use_aligned_float8,
                                   int64_t         ne00,
                                   int64_t         stride_token,
                                   int64_t         ne0,
                                   int64_t         n_tokens,
                                   int64_t         nrows_dst,
                                   int             n_expert_used,
                                   cudaStream_t    stream);

void quantize_scatter_mmq_q8_1_cuda(const float *   x,
                                    const int32_t * ids_src1_inv,
                                    void *          vy,
                                    ggml_type       type_src0,
                                    int64_t         ne00,
                                    int64_t         stride_token,
                                    int64_t         ne0,
                                    int64_t         n_tokens,
                                    int64_t         nrows_dst,
                                    int             n_expert_used,
                                    cudaStream_t    stream);
