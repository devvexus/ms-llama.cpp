#pragma once

#include "common.cuh"

// batched MS_C x f32 matrix multiplication on tensor cores (the batch sizes after MMVQ, and the small ones MMVQ is slow at)
bool ggml_cuda_should_use_mmms(enum ggml_type type, int cc, int64_t ne11);

void ggml_cuda_mul_mat_ms(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// MUL_MAT_ID of MS_C experts on tensor cores (the batches after MMVQ): the (token, slot) pairs of each expert in column tiles
bool ggml_cuda_should_use_mmms_id(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_ms_id(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst);

// FFN gate and up of the same MS_C type in one kernel that writes glu(gate, up) (prompt batches)
bool ggml_cuda_should_fuse_mul_mat_ms(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * glu, int cc);

void ggml_cuda_mul_mat_ms_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * up, const ggml_tensor * gate, ggml_tensor * glu);
