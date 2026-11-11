#pragma once

#include "common.cuh"

// batched MS_C x f32 matrix multiplication on tensor cores (the batch sizes after MMVQ)
bool ggml_cuda_should_use_mmms(enum ggml_type type, int cc, int64_t ne11);

void ggml_cuda_mul_mat_ms(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
