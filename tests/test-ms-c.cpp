// MS_C CPU kernels (GGML_TYPE_MS_C1 .. GGML_TYPE_MS_C6) against a scalar implementation of the format in this file:
//  - to_float, and vec_dot with F32 activations
//  - MUL_MAT through the CPU backend with the weights in a CPU buffer (F32 activations)
//  - MUL_MAT with the weights in the CPU_REPACK buffer, where the build has MS_C paths there (int8 activations on ARM, int16
//    activations on x86), against references of the same integer math
//  - the same results with 1, 4 and all threads, and for two mat-vecs of one column in one graph

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

static const int QK = 512; // weights per block

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { if (n_fail++ < 20) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } } while (0)

static std::mt19937 rng(1234);

static int nths[3]; // 1, 4 and all threads

static float frand(float a, float b) {
    return std::uniform_real_distribution<float>(a, b)(rng);
}

static ggml_type ms_type(int k) {
    return (ggml_type) (GGML_TYPE_MS_C1 + k - 1);
}

static size_t ms_block_size(int k) {
    return 4 + (size_t) 64*k;
}

static int ms_value(uint32_t s) {
    const uint32_t x = 34038481u*s + 76625530u;
    return (int) ((x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24)) - 510;
}

// a block is the FP16 scale d, s0 (its low 14 bits are the first state), then 511 symbols of k bits (MSB first) and k pad bits;
// the next state is ((s << k) | symbol) & 0x3FFF, and the weight of state s is ms_value(s)*d
static float ms_decode_block(int k, const uint8_t * blk, int * v) {
    ggml_fp16_t dh;
    uint16_t    s0;
    memcpy(&dh, blk, sizeof(dh));
    memcpy(&s0, blk + 2, sizeof(s0));

    uint32_t s   = s0 & 0x3FFF;
    size_t   bit = 0;
    v[0] = ms_value(s);
    for (int i = 1; i < QK; i++) {
        uint32_t b = 0;
        for (int j = 0; j < k; j++, bit++) {
            b = (b << 1) | ((blk[4 + bit/8] >> (7 - bit%8)) & 1);
        }
        s = ((s << k) | b) & 0x3FFF;
        v[i] = ms_value(s);
    }
    return ggml_fp16_to_fp32(dh);
}

struct ms_weights {
    int                  k;
    int64_t              K;
    int64_t              M;
    std::vector<uint8_t> data; // M rows of K/QK blocks
    std::vector<int>     v;    // M*K values
    std::vector<float>   d;    // M*(K/QK) scales
};

enum scale_mode {
    SCALE_MIXED, // zero, FP16 subnormals, many magnitudes up to 2047, both signs
    SCALE_POW2,  // one power of two per row: with small integer activations all partial sums are exact in float
    SCALE_LARGE, // SCALE_MIXED, and 3000 in some blocks (32*d does not fit in FP16)
};

static float make_scale(scale_mode mode, int64_t r, int64_t b) {
    static const float mag[] = { 0.0f, 3.0e-6f, 6.1e-5f, 1.0e-3f, 2.0e-2f, 0.25f, 7.5f, 300.0f, 2047.0f };
    if (mode == SCALE_POW2) {
        return ldexpf(r % 2 ? -1.0f : 1.0f, (int) (r % 12) - 10);
    }
    if (mode == SCALE_LARGE && r % 5 == 2 && b % 2 == 1) {
        return 3000.0f;
    }
    const float m = mag[(r*5 + b*3 + 4) % 9];
    return (r + b) % 2 ? -m : m;
}

// the kind of a block is (r + 2*b) % 6: random symbols, s0 with its top bits set, zero pad bits, all symbols 0, all symbols 1,
// or the symbols that give the largest |v| at each step
static ms_weights make_weights(int k, int64_t K, int64_t M, scale_mode mode) {
    ms_weights w = { k, K, M, {}, {}, {} };

    const int64_t nb = K/QK;
    const size_t  bs = ms_block_size(k);
    w.data.resize(M*nb*bs);
    w.v.resize(M*K);
    w.d.resize(M*nb);

    for (int64_t r = 0; r < M; r++) {
        for (int64_t b = 0; b < nb; b++) {
            uint8_t * blk = w.data.data() + (r*nb + b)*bs;
            const int kind = (int) ((r + 2*b) % 6);

            const ggml_fp16_t dh = ggml_fp32_to_fp16(make_scale(mode, r, b));
            uint16_t s0 = (uint16_t) (rng() & 0x3FFF);
            if (kind == 1) {
                s0 |= (uint16_t) (0x8000 | (rng() & 0x4000));
            }
            memcpy(blk, &dh, sizeof(dh));
            memcpy(blk + 2, &s0, sizeof(s0));

            for (size_t j = 4; j < bs; j++) {
                blk[j] = kind == 3 ? 0x00 : kind == 4 ? 0xFF : (uint8_t) rng();
            }
            if (kind == 2) {
                blk[bs - 1] &= (uint8_t) (0xFF << k);
            }
            if (kind == 5) {
                memset(blk + 4, 0, bs - 4);
                uint32_t s   = s0 & 0x3FFF;
                size_t   bit = 0;
                for (int i = 1; i < QK; i++) {
                    uint32_t best = 0;
                    for (uint32_t c = 1; c < (1u << k); c++) {
                        if (std::abs(ms_value(((s << k) | c) & 0x3FFF)) > std::abs(ms_value(((s << k) | best) & 0x3FFF))) {
                            best = c;
                        }
                    }
                    for (int j = k - 1; j >= 0; j--, bit++) {
                        blk[4 + bit/8] |= (uint8_t) (((best >> j) & 1) << (7 - bit%8));
                    }
                    s = ((s << k) | best) & 0x3FFF;
                }
            }
            w.d[r*nb + b] = ms_decode_block(k, blk, w.v.data() + r*K + b*QK);
        }
    }
    return w;
}

// N columns of K activations. mode 0: by 512 values, uniform with outliers, all zero, all +-1 (+-127 or +-8191 after
// quantization), or one spike in each 32 values; mode 1: integers in [-3, 3] times a power of two per column
static std::vector<float> make_x(int64_t K, int64_t N, int mode) {
    std::vector<float> x(K*N);
    for (int64_t c = 0; c < N; c++) {
        for (int64_t i = 0; i < K; i++) {
            float & xi = x[c*K + i];
            if (mode == 1) {
                xi = ldexpf((float) ((int) (rng() % 7) - 3), (int) (c % 5) - 2);
                continue;
            }
            switch ((i/QK + c) % 4) {
                case 0:
                    xi = frand(-1.0f, 1.0f);
                    if (rng() % 97 == 0) {
                        xi *= 20.0f;
                    }
                    break;
                case 1:
                    xi = 0.0f;
                    break;
                case 2:
                    xi = i % 2 ? -1.0f : 1.0f;
                    break;
                default:
                    xi = i % 32 == c % 32 ? 30.0f : frand(-1e-3f, 1e-3f);
                    break;
            }
        }
    }
    return x;
}

// bound on the error of a float sum of m terms in any order, relative to the sum of their magnitudes: m*u/(1 - m*u)
static double err_bound(int64_t m) {
    const double u = ldexp(1.0, -24);
    return m*u/(1.0 - m*u);
}

// expected outputs [N][M]: bitwise y (or y_alt, if not empty) where err is 0, else within err of y
struct expected {
    std::vector<double> y;
    std::vector<double> y_alt;
    std::vector<double> err;
};

// c + a*b with the product rounded first (binary64 holds the exact product, and rounds the sum of two floats correctly)
static float add_product_unfused(float c, float a, float b) {
    const float p = (float) ((double) a*b);
    return (float) ((double) c + p);
}

static bool same_float(float a, float b) {
    return memcmp(&a, &b, sizeof(a)) == 0 || (a == 0.0f && b == 0.0f);
}

static bool same(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

// outputs that differ from e; worst gets the largest |out - y|/err (it stays below 0 if all outputs are bitwise)
static int64_t count_bad(const std::vector<float> & out, const expected & e, double & worst) {
    int64_t bad = 0;
    for (size_t i = 0; i < out.size(); i++) {
        if (e.err[i] == 0.0) {
            bad += !same_float(out[i], (float) e.y[i]) && (e.y_alt.empty() || !same_float(out[i], (float) e.y_alt[i]));
        } else {
            const double diff = std::fabs((double) out[i] - e.y[i]);
            bad += !(diff <= e.err[i]);
            worst = std::max(worst, diff/e.err[i]);
        }
    }
    return bad;
}

// F32 activations: each product v*d*x is rounded at most twice, with K - 1 sums in any order (and one more u for the double
// sums here); with exact, every partial sum of every order is exact in float
static expected ref_f32(const ms_weights & w, const std::vector<float> & x, int64_t N, bool exact) {
    const int64_t K = w.K, M = w.M;
    expected e;
    e.y.resize(N*M);
    e.err.resize(N*M);
    for (int64_t c = 0; c < N; c++) {
        for (int64_t r = 0; r < M; r++) {
            double s = 0.0, a = 0.0;
            for (int64_t i = 0; i < K; i++) {
                const double t = (double) w.v[r*K + i]*w.d[r*(K/QK) + i/QK]*x[c*K + i];
                s += t;
                a += std::fabs(t);
            }
            e.y[c*M + r]   = s;
            e.err[c*M + r] = exact ? 0.0 : err_bound(K + 2)*a;
        }
    }
    return e;
}

// the int8 mat-vec (one column, ARM): x in blocks of 32 with d = amax/127 and q = x/d rounded to nearest even; per block the exact
// integer sum I of v*q, F[b % 4] = fma(I, d_b, F[b % 4]) over the 16 blocks of an MS_C block, s += d_w*((F0 + F1) + (F2 + F3))
// with or without a fused multiply-add (the compiler decides)
static expected ref_mv_q8(const ms_weights & w, const std::vector<float> & x) {
    const int64_t K = w.K, M = w.M;
    std::vector<int8_t> q(K);
    std::vector<float>  dq(K/32);
    for (int64_t b = 0; b < K/32; b++) {
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) {
            amax = std::max(amax, std::fabs(x[b*32 + j]));
        }
        const float d  = amax/127.0f;
        const float id = d != 0.0f ? 1.0f/d : 0.0f;
        for (int j = 0; j < 32; j++) {
            q[b*32 + j] = (int8_t) std::nearbyint(x[b*32 + j]*id);
        }
        dq[b] = d;
    }

    expected e;
    e.y.resize(M);
    e.y_alt.resize(M);
    e.err.assign(M, 0.0);
    for (int64_t r = 0; r < M; r++) {
        float s = 0.0f, s_alt = 0.0f;
        for (int64_t wb = 0; wb < K/QK; wb++) {
            float F[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            for (int b = 0; b < QK/32; b++) {
                const int64_t j0 = wb*QK + b*32;
                int32_t I = 0;
                for (int i = 0; i < 32; i++) {
                    I += w.v[r*K + j0 + i]*q[j0 + i];
                }
                F[b % 4] = std::fma((float) I, dq[j0/32], F[b % 4]);
            }
            const float sumb = (F[0] + F[1]) + (F[2] + F[3]);
            s     = std::fma(w.d[r*(K/QK) + wb], sumb, s);
            s_alt = add_product_unfused(s_alt, w.d[r*(K/QK) + wb], sumb);
        }
        e.y[r]     = s;
        e.y_alt[r] = s_alt;
    }
    return e;
}

// the int8 GEMM (ARM): x as Q8_0 (blocks of 32, d = amax/127 stored as FP16, q = x/d rounded to nearest even as in the NEON
// quantizers), v = 32*hi + lo with hi in [-16, 15] and lo in [0, 31]. With i8mm (exact): over the hi blocks, then the lo blocks,
// s = fma(I, p, s) with the exact integer sum I of a block and p = (32*d_w or d_w)*d_x. Without i8mm the sum of the 2*K/32 terms
// I*p in another order: each term rounded at most twice, with 2*K/32 - 1 sums in any order (and one u for the double sums here)
static expected ref_gemm_q8_0(const ms_weights & w, const std::vector<float> & x, int64_t N, bool exact) {
    const int64_t K = w.K, M = w.M, nb = K/32;
    std::vector<int8_t> q(K);
    std::vector<float>  dq(nb);

    expected e;
    e.y.resize(N*M);
    e.err.resize(N*M);
    for (int64_t c = 0; c < N; c++) {
        const float * xc = x.data() + c*K;
        for (int64_t b = 0; b < nb; b++) {
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) {
                amax = std::max(amax, std::fabs(xc[b*32 + j]));
            }
            const float d  = amax/127.0f;
            const float id = d != 0.0f ? 1.0f/d : 0.0f;
            for (int j = 0; j < 32; j++) {
                q[b*32 + j] = (int8_t) std::nearbyint(xc[b*32 + j]*id);
            }
            dq[b] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d));
        }
        for (int64_t r = 0; r < M; r++) {
            float  s  = 0.0f;
            double sd = 0.0, a = 0.0;
            for (int64_t pb = 0; pb < 2*nb; pb++) {
                const int64_t j = pb % nb;
                int32_t I = 0;
                for (int i = 0; i < 32; i++) {
                    const int u = w.v[r*K + j*32 + i] + 512;
                    I += (pb < nb ? (u >> 5) - 16 : u & 31)*q[j*32 + i];
                }
                const float dw = w.d[r*(K/QK) + j/(QK/32)];
                const float p  = (pb < nb ? 32.0f*dw : dw)*dq[j];
                s   = std::fma((float) I, p, s);
                sd += (double) I*p;
                a  += std::fabs((double) I*p);
            }
            e.y[c*M + r]   = exact ? s : sd;
            e.err[c*M + r] = exact ? 0.0 : err_bound(2*nb + 2)*a;
        }
    }
    return e;
}

// the int16 GEMM (x86): x in blocks of 512 with s = amax/8191 and q = x*(8191/amax) rounded to nearest even; per block the exact
// integer sum I of v*q, and over the blocks t = fma(I, d*s, t)
static expected ref_gemm_i16(const ms_weights & w, const std::vector<float> & x, int64_t N) {
    const int64_t K = w.K, M = w.M, nb = K/QK;
    std::vector<int16_t> q(K);
    std::vector<float>   sq(nb);

    expected e;
    e.y.resize(N*M);
    e.err.assign(N*M, 0.0);
    for (int64_t c = 0; c < N; c++) {
        const float * xc = x.data() + c*K;
        for (int64_t b = 0; b < nb; b++) {
            float amax = 0.0f;
            for (int j = 0; j < QK; j++) {
                amax = std::max(amax, std::fabs(xc[b*QK + j]));
            }
            const float id = amax > 8191.0f/FLT_MAX ? 8191.0f/amax : 0.0f;
            for (int j = 0; j < QK; j++) {
                q[b*QK + j] = (int16_t) std::nearbyint(xc[b*QK + j]*id);
            }
            sq[b] = amax/8191.0f;
        }
        for (int64_t r = 0; r < M; r++) {
            float t = 0.0f;
            for (int64_t b = 0; b < nb; b++) {
                int64_t I = 0;
                for (int i = 0; i < QK; i++) {
                    I += w.v[r*K + b*QK + i]*q[b*QK + i];
                }
                t = std::fma((float) I, w.d[r*nb + b]*sq[b], t);
            }
            e.y[c*M + r] = t;
        }
    }
    return e;
}

static std::vector<float> run_vec_dot(const ms_weights & w, const std::vector<float> & x, int64_t N) {
    const ggml_type_traits_cpu * tc = ggml_get_type_traits_cpu(ms_type(w.k));
    const size_t rs = ggml_row_size(ms_type(w.k), w.K);
    std::vector<float> out(N*w.M);
    for (int64_t c = 0; c < N; c++) {
        for (int64_t r = 0; r < w.M; r++) {
            tc->vec_dot((int) w.K, &out[c*w.M + r], 0, w.data.data() + r*rs, 0, x.data() + c*w.K, 0, 1);
        }
    }
    return out;
}

// MUL_MAT of each weight with one x, all in one graph (two weights and one column: the pair of mat-vecs)
struct mm_graph {
    ggml_backend_t             backend;
    ggml_context *             ctx_w = nullptr;
    ggml_context *             ctx   = nullptr;
    ggml_backend_buffer_t      buf_w = nullptr;
    ggml_backend_buffer_t      buf   = nullptr;
    ggml_cgraph *              graph = nullptr;
    std::vector<ggml_tensor *> outs;

    mm_graph(ggml_backend_t be, ggml_backend_buffer_type_t buft, const std::vector<const ms_weights *> & ws,
             const std::vector<float> & x, int64_t N) : backend(be) {
        const ggml_init_params ip = { 8*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ctx_w = ggml_init(ip);
        ctx   = ggml_init(ip);

        std::vector<ggml_tensor *> W;
        for (const ms_weights * w : ws) {
            W.push_back(ggml_new_tensor_2d(ctx_w, ms_type(w->k), w->K, w->M));
        }
        buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
        for (size_t i = 0; i < W.size(); i++) {
            ggml_backend_tensor_set(W[i], ws[i]->data.data(), 0, ggml_nbytes(W[i]));
        }

        ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ws[0]->K, N);
        graph = ggml_new_graph(ctx);
        for (ggml_tensor * t : W) {
            outs.push_back(ggml_mul_mat(ctx, t, X));
            ggml_build_forward_expand(graph, outs.back());
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_tensor_set(X, x.data(), 0, ggml_nbytes(X));
    }

    ~mm_graph() {
        ggml_backend_buffer_free(buf);
        ggml_backend_buffer_free(buf_w);
        ggml_free(ctx);
        ggml_free(ctx_w);
    }

    std::vector<std::vector<float>> compute(int nth) {
        std::vector<std::vector<float>> res;
        for (ggml_tensor * t : outs) {
            res.emplace_back(ggml_nelements(t), NAN);
            ggml_backend_tensor_set(t, res.back().data(), 0, ggml_nbytes(t));
        }
        ggml_backend_cpu_set_n_threads(backend, nth);
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute failed");
        for (size_t i = 0; i < outs.size(); i++) {
            ggml_backend_tensor_get(outs[i], res[i].data(), 0, ggml_nbytes(outs[i]));
        }
        return res;
    }
};

static ggml_backend_buffer_type_t find_repack_buft() {
    ggml_backend_reg_t reg = ggml_backend_cpu_reg();
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    auto get_extra_bufts = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    for (ggml_backend_buffer_type_t * b = get_extra_bufts ? get_extra_bufts(dev) : nullptr; b && *b; b++) {
        if (strcmp(ggml_backend_buft_name(*b), "CPU_REPACK") == 0) {
            return *b;
        }
    }
    return nullptr;
}

// whether the CPU backend runs a MUL_MAT with an MS_C weight in buft
static bool repack_supports(ggml_backend_t be, ggml_backend_buffer_type_t buft, int k) {
    const ggml_init_params ip = { 4*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_w = ggml_init(ip);
    ggml_context * ctx   = ggml_init(ip);
    ggml_tensor * W = ggml_new_tensor_2d(ctx_w, ms_type(k), 2048, 16);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
    ggml_tensor * Y = ggml_mul_mat(ctx, W, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 16));
    const bool ok = ggml_backend_supports_op(be, Y);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx);
    ggml_free(ctx_w);
    return ok;
}

enum mm_path {
    PATH_F32,       // the default MUL_MAT (F32 activations)
    PATH_MV_Q8,     // int8 mat-vec
    PATH_GEMM_HL,   // int8 GEMM on the hi/lo planes of 8 rows (i8mm)
    PATH_GEMM_Q8_0, // int8 GEMM with the Q8_0 4x4 kernel (dotprod)
    PATH_GEMM_I16,  // int16 GEMM
    PATH_COUNT,
};

static const char * path_names[PATH_COUNT] = { "f32", "int8 mat-vec", "int8 hi/lo GEMM", "int8 Q8_0 GEMM", "int16 GEMM" };

// the MS_C paths of the CPU_REPACK buffer (ggml_repack_get_optimal_repack_type and the MS_C tensor traits in repack.cpp): with
// AVX2 the int16 GEMM for 4 or more columns; on ARM with i8mm or dotprod the int8 mat-vec for one column and the int8 GEMM for
// 16 or more columns; else the default MUL_MAT
static mm_path repack_path(int64_t N) {
    if (ggml_cpu_has_avx2()) {
        return N >= 4 ? PATH_GEMM_I16 : PATH_F32;
    }
    if (ggml_cpu_has_neon() && (ggml_cpu_has_matmul_int8() || ggml_cpu_has_dotprod())) {
        if (N == 1) {
            return PATH_MV_Q8;
        }
        if (N >= 16) {
            return ggml_cpu_has_matmul_int8() ? PATH_GEMM_HL : PATH_GEMM_Q8_0;
        }
    }
    return PATH_F32;
}

static bool repack_has_ms_c_path() {
    return ggml_cpu_has_avx2() || (ggml_cpu_has_neon() && (ggml_cpu_has_matmul_int8() || ggml_cpu_has_dotprod()));
}

static void test_to_float(int k) {
    const ggml_type_traits * tt = ggml_get_type_traits(ms_type(k));
    CHECK(tt->blck_size == QK && tt->type_size == ms_block_size(k) && tt->to_float, "MS_C%d: type traits", k);
    if (!tt->to_float) {
        return;
    }

    const ms_weights w = make_weights(k, 6144, 6, SCALE_LARGE);
    int64_t bad = 0, n = 0;
    for (int64_t nb : { (int64_t) 1, (int64_t) 3, 6*w.K/QK }) {
        std::vector<float> y(nb*QK);
        tt->to_float(w.data.data(), y.data(), nb*QK);
        for (int64_t i = 0; i < nb*QK; i++) {
            bad += !same_float(y[i], (float) w.v[i]*w.d[i/QK]);
        }
        n += nb*QK;
    }
    CHECK(bad == 0, "MS_C%d: to_float: %lld of %lld values differ", k, (long long) bad, (long long) n);
    printf("  to_float: %lld values bitwise\n", (long long) n);
}

static void test_vec_dot(int k) {
    const ggml_type_traits_cpu * tc = ggml_get_type_traits_cpu(ms_type(k));
    CHECK(tc->vec_dot && tc->vec_dot_type == GGML_TYPE_F32, "MS_C%d: CPU type traits", k);
    if (!tc->vec_dot) {
        return;
    }

    // one-hot activations of +-2^e: exactly the weight times x_i
    const ms_weights w = make_weights(k, 2048, 2, SCALE_MIXED);
    const size_t rs = ggml_row_size(ms_type(k), w.K);
    std::vector<float> x(w.K, 0.0f);
    int64_t bad = 0;
    for (int64_t r = 0; r < w.M; r++) {
        for (int64_t i = 0; i < w.K; i++) {
            const float xi = ldexpf(i % 2 ? -1.0f : 1.0f, (int) (i % 7) - 3);
            float out;
            x[i] = xi;
            tc->vec_dot((int) w.K, &out, 0, w.data.data() + r*rs, 0, x.data(), 0, 1);
            x[i] = 0.0f;
            const float wi = (float) w.v[r*w.K + i]*w.d[r*(w.K/QK) + i/QK];
            bad += !same_float(out, wi*xi);
        }
    }
    CHECK(bad == 0, "MS_C%d: vec_dot with one-hot activations: %lld of %lld differ", k, (long long) bad, (long long) (w.M*w.K));

    // exact sums (bitwise) and random activations (within the bound), 1, 4 and 12 blocks
    double worst = -1.0;
    int64_t bad2 = 0, n2 = 0;
    for (int64_t K : { 512, 2048, 6144 }) {
        for (int exact = 0; exact < 2; exact++) {
            const ms_weights wk = make_weights(k, K, 8, exact ? SCALE_POW2 : SCALE_MIXED);
            const std::vector<float> xk = make_x(K, 4, exact);
            bad2 += count_bad(run_vec_dot(wk, xk, 4), ref_f32(wk, xk, 4, exact), worst);
            n2 += 4*wk.M;
        }
    }
    CHECK(bad2 == 0, "MS_C%d: vec_dot: %lld of %lld results differ from the reference", k, (long long) bad2, (long long) n2);
    printf("  vec_dot: %lld one-hot bitwise, %lld rows (largest error %.2g of the bound)\n", (long long) (w.M*w.K), (long long) n2, worst);
}

struct mm_shape {
    int64_t K;
    int64_t M;
};

// 1, 4 and 12 blocks per row, rows that are not a multiple of 4 or 8, and columns on both sides of each path threshold
static const mm_shape mm_shapes[] = { { 512, 1 }, { 512, 40 }, { 2048, 13 }, { 6144, 40 } };
static const int64_t  mm_cols[]   = { 1, 2, 3, 4, 5, 15, 16, 17, 33 };

static void test_mul_mat(int k, ggml_backend_t be, ggml_backend_buffer_type_t buft_repack) {
    int64_t n_cpu = 0, n_repack[PATH_COUNT] = { 0 };
    double worst_cpu = -1.0, worst_repack = -1.0;

    for (const mm_shape & sh : mm_shapes) {
        // 0: mixed scales and activations, 1: exact sums, 2: scales above 2047 in some blocks
        for (int set = 0; set < 3; set++) {
            if (set == 2 && sh.K != 2048) {
                continue;
            }
            const ms_weights w = make_weights(k, sh.K, sh.M, set == 1 ? SCALE_POW2 : set == 2 ? SCALE_LARGE : SCALE_MIXED);
            for (int64_t N : mm_cols) {
                const std::vector<float> x = make_x(sh.K, N, set == 1);

                mm_graph gc(be, ggml_backend_cpu_buffer_type(), { &w }, x, N);
                const std::vector<float> out_cpu = gc.compute(nths[2])[0];
                for (int t = 0; t < 2; t++) {
                    CHECK(same(gc.compute(nths[t])[0], out_cpu), "MS_C%d K=%lld M=%lld N=%lld: CPU buffer, %d threads differ from %d",
                          k, (long long) sh.K, (long long) sh.M, (long long) N, nths[t], nths[2]);
                }
                const int64_t bad = count_bad(out_cpu, ref_f32(w, x, N, set == 1), worst_cpu);
                CHECK(bad == 0, "MS_C%d K=%lld M=%lld N=%lld set %d: CPU buffer, %lld outputs differ from the reference",
                      k, (long long) sh.K, (long long) sh.M, (long long) N, set, (long long) bad);
                n_cpu += N*sh.M;

                if (!buft_repack) {
                    continue;
                }

                const mm_path path = repack_path(N);
                mm_graph gr(be, buft_repack, { &w }, x, N);
                const std::vector<float> out = gr.compute(nths[2])[0];
                n_repack[path] += N*sh.M;

                if (set == 2 && (path == PATH_GEMM_HL || path == PATH_GEMM_Q8_0)) {
                    // a range of rows with a scale above 2047 uses the F32 vec_dot; the ranges depend on the timing of the
                    // threads, so each row matches one of the two, and a row with such a scale matches vec_dot
                    const expected e = ref_gemm_q8_0(w, x, N, path == PATH_GEMM_HL);
                    const std::vector<float> vd = run_vec_dot(w, x, N);
                    int64_t bad_rows = 0;
                    for (int64_t r = 0; r < sh.M; r++) {
                        bool large = false;
                        for (int64_t b = 0; b < sh.K/QK; b++) {
                            large = large || std::fabs(w.d[r*(sh.K/QK) + b]) > 2047.0f;
                        }
                        bool is_vd = true, is_ref = true;
                        for (int64_t c = 0; c < N; c++) {
                            const size_t i = c*sh.M + r;
                            is_vd  = is_vd && same_float(out[i], vd[i]);
                            is_ref = is_ref && (e.err[i] == 0.0 ? same_float(out[i], (float) e.y[i]) : std::fabs(out[i] - e.y[i]) <= e.err[i]);
                        }
                        bad_rows += !(is_vd || (is_ref && !large));
                    }
                    CHECK(bad_rows == 0, "MS_C%d K=%lld M=%lld N=%lld: CPU_REPACK with scales above 2047, %lld rows differ",
                          k, (long long) sh.K, (long long) sh.M, (long long) N, (long long) bad_rows);
                    continue;
                }

                for (int t = 0; t < 2; t++) {
                    CHECK(same(gr.compute(nths[t])[0], out), "MS_C%d K=%lld M=%lld N=%lld: CPU_REPACK, %d threads differ from %d",
                          k, (long long) sh.K, (long long) sh.M, (long long) N, nths[t], nths[2]);
                }

                expected e;
                switch (path) {
                    case PATH_F32:
                        e.y.assign(out_cpu.begin(), out_cpu.end());
                        e.err.assign(out_cpu.size(), 0.0);
                        break;
                    case PATH_MV_Q8:     e = ref_mv_q8(w, x);                break;
                    case PATH_GEMM_HL:   e = ref_gemm_q8_0(w, x, N, true);   break;
                    case PATH_GEMM_Q8_0: e = ref_gemm_q8_0(w, x, N, false);  break;
                    default:             e = ref_gemm_i16(w, x, N);          break;
                }
                const int64_t bad_r = count_bad(out, e, worst_repack);
                CHECK(bad_r == 0, "MS_C%d K=%lld M=%lld N=%lld set %d: CPU_REPACK (%s), %lld outputs differ from the reference",
                      k, (long long) sh.K, (long long) sh.M, (long long) N, set, path_names[path], (long long) bad_r);
            }
        }
    }

    printf("  MUL_MAT, CPU buffer: %lld outputs (largest error %.2g of the bound)\n", (long long) n_cpu, worst_cpu);
    if (buft_repack) {
        printf("  MUL_MAT, CPU_REPACK:");
        for (int p = 0; p < PATH_COUNT; p++) {
            if (n_repack[p]) {
                printf(" %s %lld,", path_names[p], (long long) n_repack[p]);
            }
        }
        if (worst_repack < 0.0) {
            printf(" outputs, all bitwise\n");
        } else {
            printf(" outputs (largest error %.2g of the bound)\n", worst_repack);
        }
    }
}

// two mat-vecs of one column in one graph (computed together with more than one thread) give the results of each alone
static void test_pairs(int k, ggml_backend_t be, ggml_backend_buffer_type_t buft_repack) {
    const int k2 = k % 6 + 1;
    const ms_weights wa = make_weights(k, 2048, 40, SCALE_MIXED);
    const ms_weights wb = make_weights(k2, 2048, 13, SCALE_MIXED);
    const std::vector<float> x = make_x(2048, 1, 0);
    const int n_fail0 = n_fail;

    for (ggml_backend_buffer_type_t buft : { ggml_backend_cpu_buffer_type(), buft_repack }) {
        if (!buft) {
            continue;
        }
        mm_graph ga(be, buft, { &wa }, x, 1);
        mm_graph gb(be, buft, { &wb }, x, 1);
        mm_graph gp(be, buft, { &wa, &wb }, x, 1);
        for (int t = 1; t < 3; t++) {
            const std::vector<std::vector<float>> yp = gp.compute(nths[t]);
            CHECK(same(yp[0], ga.compute(nths[t])[0]) && same(yp[1], gb.compute(nths[t])[0]),
                  "MS_C%d + MS_C%d: %s, %d threads: the pair differs from the single mat-vecs", k, k2, ggml_backend_buft_name(buft), nths[t]);
        }
    }
    printf("  pair with MS_C%d: %s\n", k2, n_fail == n_fail0 ? "same results as alone" : "FAILED");
}

// the CPU_REPACK buffer logs each tensor at the debug level
static void log_no_debug(ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (level != GGML_LOG_LEVEL_DEBUG) {
        fputs(text, stderr);
    }
}

int main(void) {
    ggml_log_set(log_no_debug, nullptr);
    ggml_cpu_init();

    nths[0] = 1;
    nths[1] = 4;
    nths[2] = (int) std::max(1u, std::thread::hardware_concurrency());

    ggml_backend_t be = ggml_backend_cpu_init();

    ggml_backend_buffer_type_t buft_repack = find_repack_buft();
    if (!buft_repack) {
        printf("CPU_REPACK: not in this build, skipped\n");
    } else if (!repack_supports(be, buft_repack, 1)) {
        printf("CPU_REPACK: no MS_C path for this CPU build, skipped\n");
        buft_repack = nullptr;
    } else {
        CHECK(repack_has_ms_c_path(), "CPU_REPACK takes MS_C weights, but this test has no reference for its path");
        printf("CPU_REPACK paths of this build:");
        for (int64_t N : mm_cols) {
            printf(" N=%lld %s,", (long long) N, path_names[repack_path(N)]);
        }
        printf(" the other MS_C paths are for other CPUs\n");
    }
    printf("threads: %d, %d, %d\n", nths[0], nths[1], nths[2]);

    for (int k = 1; k <= 6; k++) {
        printf("MS_C%d\n", k);
        if (buft_repack) {
            CHECK(repack_supports(be, buft_repack, k), "MS_C%d: CPU_REPACK does not take this type", k);
        }
        test_to_float(k);
        test_vec_dot(k);
        test_mul_mat(k, be, buft_repack);
        test_pairs(k, be, buft_repack);
    }

    ggml_backend_free(be);

    if (n_fail) {
        printf("%d checks failed\n", n_fail);
    } else {
        printf("all checks passed\n");
    }
    return n_fail != 0;
}
