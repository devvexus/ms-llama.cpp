#include "mmms.cuh"
#include "mmvq.cuh"
#include "mmid.cuh"
#include "unary.cuh"
#include "cp-async.cuh"
#include "graph.cuh"

// Batched MS_C x f32 matrix multiplication on tensor cores (mma.m16n8k16, f16 inputs, f32 sums).
//
// The MS_C blocks are decoded in registers straight into the A fragments of the mma, there is no f16 copy of the weights.
// The A and B values are the ones of the dequantize + cuBLAS path: w = f16(v*d) and f16(y).
//
// The products of an mma can be summed in any order of k, so each thread decodes runs of 16 consecutive weights,
// and y is stored in the matching order (mmms_convert_y): within each group of MMMS_BK = 64 k values,
// k value p = 16*t + 4*s + 2*h + e goes to position 16*s + 2*t + 8*h + e,
// the position from which mma step s reads it in the fragment half h of the threads with lane % 4 == t.

#define MMMS_BK   64 // k values per pipeline stage
#define MMMS_SROW (MMMS_BK + 8) // f16 values per smem row of the y tile, the padding avoids bank conflicts of ldmatrix

// Turing has no cp.async: the y tiles are copied synchronously, the barrier of each chunk makes them visible
static __device__ __forceinline__ void mmms_cp_async_commit() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;");
#endif // CP_ASYNC_AVAILABLE
}

template <int n>
static __device__ __forceinline__ void mmms_cp_async_wait() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group %0;" : : "n"(n));
#endif // CP_ASYNC_AVAILABLE
}

// async: cp.async (if available); the synchronous copy is faster for the narrow n tiles of small batches
template <bool async>
static __device__ __forceinline__ void mmms_copy_16(const unsigned int dst, const void * src) {
#ifdef CP_ASYNC_AVAILABLE
    if constexpr (async) {
        cp_async_cg_16<0>(dst, src);
        return;
    }
#endif // CP_ASYNC_AVAILABLE
    const uint4 v = *(const uint4 *) src;
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" : : "r"(dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

// D (+)= A*B for one m16n8k16 tile with f16 sums (first: D = A*B), on Turing as two m16n8k8
template <bool first>
static __device__ __forceinline__ void mmms_mma_f16(uint32_t & d0, uint32_t & d1,
        const uint32_t a0, const uint32_t a1, const uint32_t a2, const uint32_t a3, const uint32_t b0, const uint32_t b1) {
#if defined(AMPERE_MMA_AVAILABLE)
    if constexpr (first) {
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %8};"
            : "=r"(d0), "=r"(d1) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0));
    } else {
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
            : "+r"(d0), "+r"(d1) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
#elif defined(TURING_MMA_AVAILABLE)
    if constexpr (first) {
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%5, %5};"
            : "=r"(d0), "=r"(d1) : "r"(a0), "r"(a1), "r"(b0), "r"(0));
    } else {
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(d0), "+r"(d1) : "r"(a0), "r"(a1), "r"(b0));
    }
    asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
        : "+r"(d0), "+r"(d1) : "r"(a2), "r"(a3), "r"(b1));
#else
    GGML_UNUSED_VARS(d0, d1, a0, a1, a2, a3, b0, b1);
    NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE)
}

// D += A*B for one m16n8k16 tile with f32 sums, on Turing as two m16n8k8
static __device__ __forceinline__ void mmms_mma_f32(float (&d)[4],
        const uint32_t a0, const uint32_t a1, const uint32_t a2, const uint32_t a3, const uint32_t b0, const uint32_t b1) {
#if defined(AMPERE_MMA_AVAILABLE)
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif defined(TURING_MMA_AVAILABLE)
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(b0));
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a2), "r"(a3), "r"(b1));
#else
    GGML_UNUSED_VARS(d, a0, a1, a2, a3, b0, b1);
    NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE)
}

// y as f16 in the k order above: [ne12*ne13][ne11_pad][ne10], the columns ne11..ne11_pad-1 are zero.
// Each column is scaled by a power of two so that its largest magnitude is in [32, 64): the f16 values and the f16 sums
// of the mma do not overflow, and the scale is exact (inv_scale undoes it on the output).
// One block per column; a thread writes 16 bytes of y, the words 4*q .. 4*q+3 (q = 2*s + h) of a group of MMMS_BK
// k values: word t holds the k values 16*t + 4*s + 2*h + e (e = 0, 1).
#define MMMS_CONVERT_NT 256
template <bool aligned>
static __global__ void __launch_bounds__(MMMS_CONVERT_NT)
mmms_convert_y(
        const float * __restrict__ x, uint4 * __restrict__ y, float * __restrict__ inv_scale,
        const int ne10, const int ne11, const int ne11_pad, const int ne12,
        const int64_t s11, const int64_t s12, const int64_t s13) {
    const int64_t col = blockIdx.x;
    const int     i11 = col % ne11_pad;
    const int     i12 = (col / ne11_pad) % ne12;
    const int64_t i13 = (col / ne11_pad) / ne12;
    const int     tid = threadIdx.x;

    uint4 * yc = y + col*(ne10/8);
    if (i11 >= ne11) {
        for (int iq = tid; iq < ne10/8; iq += MMMS_CONVERT_NT) {
            yc[iq] = make_uint4(0, 0, 0, 0);
        }
        if (tid == 0) {
            inv_scale[col] = 1.0f;
        }
        return;
    }

    const float * xc = x + i13*s13 + i12*s12 + i11*s11;

    float amax = 0.0f;
    for (int i = 2*tid; i < ne10; i += 2*MMMS_CONVERT_NT) {
        const float2 v = aligned ? *(const float2 *) (xc + i) : make_float2(xc[i], xc[i + 1]);
        amax = fmaxf(amax, fmaxf(fabsf(v.x), fabsf(v.y)));
    }
    amax = warp_reduce_max(amax);
    __shared__ float amax_warp[MMMS_CONVERT_NT/WARP_SIZE];
    if (tid % WARP_SIZE == 0) {
        amax_warp[tid / WARP_SIZE] = amax;
    }
    __syncthreads();
    amax = 0.0f;
#pragma unroll
    for (int w = 0; w < MMMS_CONVERT_NT/WARP_SIZE; ++w) {
        amax = fmaxf(amax, amax_warp[w]);
    }

    // amax = m*2^e with m in [0.5, 1): scale by 2^(6 - e), not for 0, inf or nan
    int e = 0;
    if (amax > 0.0f && isfinite(amax)) {
        frexpf(amax, &e);
        e = min(max(6 - e, -100), 100);
    } else {
        e = 0;
    }
    const float scale = ldexpf(1.0f, e/2)*ldexpf(1.0f, e - e/2); // 2^e also beyond the float exponent range of one ldexpf
    if (tid == 0) {
        inv_scale[col] = 1.0f/scale;
    }

    for (int iq = tid; iq < ne10/8; iq += MMMS_CONVERT_NT) {
        const int q = iq % (MMMS_BK/8);
        const float * xq = xc + (iq / (MMMS_BK/8))*MMMS_BK + 4*(q/2) + 2*(q%2);
        uint32_t w[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const float2 v = aligned ? *(const float2 *) (xq + 16*t) : make_float2(xq[16*t], xq[16*t + 1]);
            const half2 h = __floats2half2_rn(v.x*scale, v.y*scale);
            w[t] = *(const uint32_t *) &h;
        }
        yc[iq] = make_uint4(w[0], w[1], w[2], w[3]);
    }
}

// mmms_convert_y reading the column once: block (col) keeps it in registers, r iq groups of 8 values per thread
#define MMMS_CONVERT1_NT 512
template <bool aligned, int r>
static __global__ void __launch_bounds__(MMMS_CONVERT1_NT)
mmms_convert_y1(
        const float * __restrict__ x, uint4 * __restrict__ y, float * __restrict__ inv_scale,
        const int ne10, const int ne11, const int ne11_pad, const int ne12,
        const int64_t s11, const int64_t s12, const int64_t s13) {
    const int64_t col = blockIdx.x;
    const int     i11 = col % ne11_pad;
    const int     i12 = (col / ne11_pad) % ne12;
    const int64_t i13 = (col / ne11_pad) / ne12;
    const int     tid = threadIdx.x;
    const int     niq = ne10/8;

    uint4 * yc = y + col*niq;
    if (i11 >= ne11) {
        for (int iq = tid; iq < niq; iq += MMMS_CONVERT1_NT) {
            yc[iq] = make_uint4(0, 0, 0, 0);
        }
        if (tid == 0) {
            inv_scale[col] = 1.0f;
        }
        return;
    }

    const float * xc = x + i13*s13 + i12*s12 + i11*s11;

    float2 v[r][4];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < r; ++j) {
        const int iq = tid + j*MMMS_CONVERT1_NT;
        if (iq < niq) {
            const int q = iq % (MMMS_BK/8);
            const float * xq = xc + (iq / (MMMS_BK/8))*MMMS_BK + 4*(q/2) + 2*(q%2);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                v[j][t] = aligned ? *(const float2 *) (xq + 16*t) : make_float2(xq[16*t], xq[16*t + 1]);
                amax = fmaxf(amax, fmaxf(fabsf(v[j][t].x), fabsf(v[j][t].y)));
            }
        }
    }
    amax = warp_reduce_max(amax);
    __shared__ float amax_warp[MMMS_CONVERT1_NT/WARP_SIZE];
    if (tid % WARP_SIZE == 0) {
        amax_warp[tid / WARP_SIZE] = amax;
    }
    __syncthreads();
    amax = 0.0f;
#pragma unroll
    for (int w = 0; w < MMMS_CONVERT1_NT/WARP_SIZE; ++w) {
        amax = fmaxf(amax, amax_warp[w]);
    }

    int e = 0;
    if (amax > 0.0f && isfinite(amax)) {
        frexpf(amax, &e);
        e = min(max(6 - e, -100), 100);
    } else {
        e = 0;
    }
    const float scale = ldexpf(1.0f, e/2)*ldexpf(1.0f, e - e/2);
    if (tid == 0) {
        inv_scale[col] = 1.0f/scale;
    }

#pragma unroll
    for (int j = 0; j < r; ++j) {
        const int iq = tid + j*MMMS_CONVERT1_NT;
        if (iq < niq) {
            uint32_t w[4];
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const half2 h = __floats2half2_rn(v[j][t].x*scale, v[j][t].y*scale);
                w[t] = *(const uint32_t *) &h;
            }
            yc[iq] = make_uint4(w[0], w[1], w[2], w[3]);
        }
    }
}

// state bits [pos, pos + 14) of the MSB-first words wn
template <int pos, int nw>
static __device__ __forceinline__ uint32_t mmms_state(const uint32_t (&wn)[nw]) {
    constexpr int i  = pos / 32;
    constexpr int sh = pos % 32;
    if constexpr (sh + GGML_MS_L <= 32) {
        return (wn[i] << sh) >> (32 - GGML_MS_L);
    } else {
        static_assert(i + 1 < nw, "state outside of the loaded words");
        return __funnelshift_l(wn[i + 1], wn[i], sh) >> (32 - GGML_MS_L);
    }
}

// f16 bits of 1536 + v for state s: 0x6600 + v is in [0x6400, 0x6800), where the f16 spacing is 1
static __device__ __forceinline__ uint32_t mmms_value_f16(const uint32_t s) {
    const uint32_t x = GGML_MS_C_A*s + GGML_MS_C_B;
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A
    return __dp4a(x, 0x01010101u, 0x6600u - 510u);
#else
    // no unsigned __dp4a (HIP defines __CUDA_ARCH__ but has no __dp4a; older NVIDIA archs): the same sum in plain integer ops
    return (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24) + 0x6600u - 510u;
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A
}

// words per block: the stream [s0][qs] is 32 + 512*k bits, the word i >= 1 is the big endian word i of the block
template <int k>
struct mmms_block {
    static constexpr int bs   = 4 + 64*k;               // bytes
    static constexpr int nwn  = (15*k + 13)/32 + 1;     // words holding the 16 states of a thread
    static constexpr int nwl  = nwn + 1;                // loaded words (the run can start anywhere in a word)
    static constexpr int wmax = 16*k;                   // last word of the stream
};

// the raw words of the 16 states that thread t decodes in chunk c (of 8) of block ib of a row
template <int k>
static __device__ __forceinline__ void mmms_load_words(
        const char * __restrict__ xr, const int ib, const int c, const int t, uint32_t (&w)[mmms_block<k>::nwl]) {
    using bl = mmms_block<k>;
    const int b0 = 32 - GGML_MS_L + k*(c*MMMS_BK + 16*t);
    const uint32_t * wp = (const uint32_t *) (xr + (int64_t) ib*bl::bs);
#pragma unroll
    for (int i = 0; i < bl::nwl; ++i) {
        w[i] = wp[min(b0/32 + i, bl::wmax)];
    }
}

// the code bits of the run of 16 states of thread t in chunk c, MSB first from the first state
template <int k>
static __device__ __forceinline__ void mmms_prep(
        const uint32_t (&w)[mmms_block<k>::nwl], const int c, const int t, uint32_t (&wn)[mmms_block<k>::nwn]) {
    using bl = mmms_block<k>;
    const int b0 = 32 - GGML_MS_L + k*(c*MMMS_BK + 16*t);

    uint32_t wb[bl::nwl];
    wb[0] = b0 < 32 ? w[0] >> 16 : __byte_perm(w[0], 0, 0x0123); // word 0 holds d (low) and s0 (high)
#pragma unroll
    for (int i = 1; i < bl::nwl; ++i) {
        wb[i] = __byte_perm(w[i], 0, 0x0123);
    }
#pragma unroll
    for (int i = 0; i < bl::nwn; ++i) {
        wn[i] = __funnelshift_l(wb[i + 1], wb[i], b0 % 32);
    }
}

// the weights f16(v*d) of states 4*s .. 4*s+3 of the run: f0 = states 4s, 4s+1, f1 = states 4s+2, 4s+3
template <int k, int s>
static __device__ __forceinline__ void mmms_decode_step(
        const uint32_t (&wn)[mmms_block<k>::nwn], const half2 d2, uint32_t & f0, uint32_t & f1) {
    using bl = mmms_block<k>;
    const uint32_t h0 = mmms_value_f16(mmms_state<k*(4*s + 0), bl::nwn>(wn));
    const uint32_t h1 = mmms_value_f16(mmms_state<k*(4*s + 1), bl::nwn>(wn));
    const uint32_t h2 = mmms_value_f16(mmms_state<k*(4*s + 2), bl::nwn>(wn));
    const uint32_t h3 = mmms_value_f16(mmms_state<k*(4*s + 3), bl::nwn>(wn));
    const half2 off = make_half2(1536.0f, 1536.0f);
    const uint32_t p0 = __byte_perm(h0, h1, 0x5410);
    const uint32_t p1 = __byte_perm(h2, h3, 0x5410);
    const half2 w0 = __hmul2(__hsub2(*(const half2 *) &p0, off), d2);
    const half2 w1 = __hmul2(__hsub2(*(const half2 *) &p1, off), d2);
    f0 = *(const uint32_t *) &w0;
    f1 = *(const uint32_t *) &w1;
}

static __device__ __forceinline__ half2 mmms_load_d(const char * __restrict__ xr, const int ib, const int bs) {
    const uint32_t w0 = *(const uint32_t *) (xr + (int64_t) ib*bs);
    const uint32_t d2 = __byte_perm(w0, 0, 0x1010);
    return *(const half2 *) &d2;
}


// Tile: 16*nwarps rows of src0 x bn columns of src1, warp tile: 16 rows x bn columns.
// Pipeline per chunk of MMMS_BK k values: the y tiles are loaded with cp.async MMMS_NSTAGES - 1 chunks ahead,
// the code words one chunk ahead, the weights of each k16 step are decoded right before its mma.
// Stream-k: the (tile, chunk) iterations are split evenly over the blocks of the grid. A tile done by one block is written
// to dst, the parts of a split tile go to a buffer (slot 0: first tile of the block, slot 1: last tile of the block)
// and mul_mat_ms_c_fixup adds them in block order.
#define MMMS_NSTAGES 3
#define MMMS_SK_MIN_CHUNKS 8 // stream-k: fewest chunks per block

template <int i, int n, typename F>
static __device__ __forceinline__ void static_for(F && f) {
    if constexpr (i < n) {
        f(std::integral_constant<int, i>{});
        static_for<i + 1, n>(f);
    }
}

template <int bn>
static constexpr size_t mmms_nbytes_shared() {
    return MMMS_NSTAGES*bn*MMMS_SROW*sizeof(half);
}

struct mmms_args {
    const char * x; const half2 * y; const float * inv_scale; float * dst; float * part;
    const char * x_up; int glu_op; float glu_limit; // gate/up fusion: x is the gate, x_up the up weights
    int64_t s01_up; int64_t s02_up; int64_t s03_up;
    int ne00; int ne01; int64_t s01; int64_t s02; int64_t s03;
    int ne1; int ne11_pad; int ne12; int ne13; int r2; int r3;
    int64_t s1; int64_t s2; int64_t s3;
    int ntiles_m; int ntiles_n; int ntiles; int nblocks;
    // MUL_MAT_ID: the column tiles of the experts (ntiles_n of them), the sorted (token, slot) pairs, y in src1 column order
    const ggml_cuda_mm_id_tile_spec * tile_spec; const int32_t * ids_dst; int n_used; int ne11_src;
};

// first iteration of block b
static __device__ __forceinline__ int64_t mmms_it_begin(const int b, const int64_t total, const int nblocks) {
    return b*total/nblocks;
}

// the dst element of value l of n tile j of the thread (row, column within the tile)
static __device__ __forceinline__ int mmms_out_row(const int warp, const int lane, const int l) {
    return 16*warp + lane/4 + 8*(l/2);
}
static __device__ __forceinline__ int mmms_out_col(const int lane, const int j, const int l) {
    return 8*j + 2*(lane%4) + l%2;
}

// glu: warps 0 .. nwarps/2 - 1 compute rows of the gate (a.x), the others the same rows of the up weights (a.x_up),
// the tile is written as glu(gate, up); one block per tile (no stream-k), bm = 8*nwarps output rows
// ids (MUL_MAT_ID): column tile tn is a.tile_spec[tn], up to bn sorted (token, slot) pairs of one expert; y holds the src1
// columns in their own order, each pair reads its src1 column and writes its dst column; one block per tile (no stream-k)
template <int k, int nwarps, int bn, bool acc_f16, bool glu, bool ids>
static __device__ __forceinline__ void mul_mat_ms_c_body(
        const mmms_args & a, const char * __restrict__ xw, const int64_t s01, const int64_t s02, const int64_t s03) {
#if defined(TURING_MMA_AVAILABLE)
    static_assert(!(glu && ids), "no gate/up fusion for MUL_MAT_ID");
    using bl = mmms_block<k>;
    constexpr int bm      = glu ? 8*nwarps : 16*nwarps;
    constexpr int nthr    = nwarps*WARP_SIZE;
    constexpr int ntn     = bn/8;                  // n tiles of the mma
    constexpr int npieces = bn*MMMS_BK*2/16;       // 16 byte pieces of a y stage
    constexpr int cpb     = QK_MS/MMMS_BK;         // chunks per block
    constexpr int sbytes  = bn*MMMS_SROW*sizeof(half);
    static_assert(ntn % 2 == 0 && npieces % nthr == 0, "bad tile");

    extern __shared__ half2 tile_y[];
    const unsigned int ty0 = (unsigned int) __cvta_generic_to_shared(tile_y);

    // ids: the y column and the (token, slot) pair of each column of the tile, after the y stages
    [[maybe_unused]] int * ids_ycol = (int *) ((char *) tile_y + MMMS_NSTAGES*sbytes);
    [[maybe_unused]] int * ids_pair = ids_ycol + bn;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp*WARP_SIZE + lane;
    const int g    = lane / 4;
    const int t    = lane % 4;

    const int     nchunks = a.ne00/MMMS_BK;
    const int64_t total   = (int64_t) a.ntiles*nchunks;
    const int64_t it_end  = mmms_it_begin(blockIdx.x + 1, total, a.nblocks);
    int64_t it  = mmms_it_begin(blockIdx.x, total, a.nblocks);
    int     seg = 0;

    while (it < it_end) {
        // tiles in the order (batch, row tile, column tile): blocks running at the same time share rows of src0
        const int tile = it / nchunks;
        const int kc0  = it % nchunks;
        const int kc1  = min((int64_t) nchunks, kc0 + (it_end - it));
        // ids: the row tiles of a column tile are neighbours (they read the same y columns)
        const int tn   = ids ? tile / a.ntiles_m : tile % a.ntiles_n;
        const int tm   = ids ? tile % a.ntiles_m : (tile / a.ntiles_n) % a.ntiles_m;
        const int iz   = ids ? 0 : tile / (a.ntiles_n*a.ntiles_m);
        const int i12  = iz % a.ne12;
        const int i13  = iz / a.ne12;

        const int col0 = ids ? 0 : tn*bn;
        const int wr   = glu ? warp % (nwarps/2) : warp; // warp row
        const int row0 = tm*bm + 16*wr;

        const char * x0 = xw + (i13/a.r3)*s03 + (i12/a.r2)*s02;
        [[maybe_unused]] int ncols = 0; // ids: pairs in the tile
        if constexpr (ids) {
            const ggml_cuda_mm_id_tile_spec ts = a.tile_spec[tn];
            if (ts.col_low == ts.col_high) {
                return; // one tile per block
            }
            ncols = ts.col_high - ts.col_low;
            for (int j = tid; j < bn; j += nthr) {
                const int pair = a.ids_dst[ts.col_low + min(j, ncols - 1)];
                const int iex  = pair % a.n_used;
                ids_ycol[j] = (pair / a.n_used)*a.ne11_src + iex % a.ne11_src;
                ids_pair[j] = pair;
            }
            __syncthreads();
            x0 = xw + (int64_t) ts.expert*s02;
        }
        const char * xa = x0 + min(row0 + g + 0, a.ne01 - 1)*s01;
        const char * xb = x0 + min(row0 + g + 8, a.ne01 - 1)*s01;

        const half2 * yz = a.y + ((int64_t) iz*a.ne11_pad + col0)*(a.ne00/2);

        auto load_y = [&](const int kc) {
            const unsigned int dst_tile = ty0 + (kc % MMMS_NSTAGES)*sbytes;
#pragma unroll
            for (int i = 0; i < npieces/nthr; ++i) {
                const int p    = i*nthr + tid;
                const int col  = p / 8;
                const int part = p % 8;
                const half2 * yc = ids ? a.y + (int64_t) ids_ycol[col]*(a.ne00/2) : yz + (int64_t) col*(a.ne00/2);
                mmms_copy_16<(bn >= 64)>(dst_tile + (col*MMMS_SROW + 8*part)*sizeof(half), yc + kc*(MMMS_BK/2) + 4*part);
            }
        };

        uint32_t wa[bl::nwl];
        uint32_t wb[bl::nwl];
        half2 da;
        half2 db;
        auto load_words = [&](const int kc) {
            const int ib = kc / cpb;
            const int c  = kc % cpb;
            mmms_load_words<k>(xa, ib, c, t, wa);
            mmms_load_words<k>(xb, ib, c, t, wb);
            da = mmms_load_d(xa, ib, bl::bs);
            db = mmms_load_d(xb, ib, bl::bs);
        };

        float acc[ntn][4];
#pragma unroll
        for (int j = 0; j < ntn; ++j) {
            acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
        }

        load_y(kc0);
        mmms_cp_async_commit();
        if (kc0 + 1 < kc1) {
            load_y(kc0 + 1);
        }
        mmms_cp_async_commit();
        load_words(kc0);

        for (int kc = kc0; kc < kc1; ++kc) {
            mmms_cp_async_wait<1>();
            __syncthreads(); // y of chunk kc is in smem, all warps are done with the stage of chunk kc - 1
            if (kc + 2 < kc1) {
                load_y(kc + 2);
            }
            mmms_cp_async_commit();

            // code bits of this chunk, then the words of the next chunk (after the last chunk: the last words again)
            uint32_t wna[bl::nwn];
            uint32_t wnb[bl::nwn];
            mmms_prep<k>(wa, kc % cpb, t, wna);
            mmms_prep<k>(wb, kc % cpb, t, wnb);
            const half2 dca = da;
            const half2 dcb = db;
            load_words(min(kc + 1, kc1 - 1));

            const unsigned int ty = ty0 + (kc % MMMS_NSTAGES)*sbytes;

            [[maybe_unused]] uint32_t hacc[ntn][2];

            // 16 warps: the weights of step s + 1 are decoded before the mma of step s are issued
            constexpr bool pipe = nwarps == 16;
            [[maybe_unused]] uint32_t an[4];
            if constexpr (pipe) {
                mmms_decode_step<k, 0>(wna, dca, an[0], an[2]);
                mmms_decode_step<k, 0>(wnb, dcb, an[1], an[3]);
            }
            static_for<0, MMMS_BK/16>([&](auto s_) {
                constexpr int s = decltype(s_)::value;
                uint32_t a0, a1, a2, a3;
                if constexpr (pipe) {
                    a0 = an[0];
                    a1 = an[1];
                    a2 = an[2];
                    a3 = an[3];
                    if constexpr (s + 1 < MMMS_BK/16) {
                        mmms_decode_step<k, s + 1>(wna, dca, an[0], an[2]);
                        mmms_decode_step<k, s + 1>(wnb, dcb, an[1], an[3]);
                    }
                } else {
                    mmms_decode_step<k, s>(wna, dca, a0, a2);
                    mmms_decode_step<k, s>(wnb, dcb, a1, a3);
                }

#pragma unroll
                for (int j = 0; j < ntn; j += 2) {
                    // matrices: (tile j, k 0..7), (tile j, k 8..15), (tile j+1, k 0..7), (tile j+1, k 8..15) of step s
                    const int mi = lane / 8;
                    const int n  = 8*(j + mi/2) + lane % 8;
                    const unsigned int addr = ty + (n*MMMS_SROW + 16*s + 8*(mi % 2))*sizeof(half);
                    uint32_t b[4];
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                        : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(addr));
#pragma unroll
                    for (int jj = 0; jj < 2; ++jj) {
                        if constexpr (acc_f16) {
                            mmms_mma_f16<s == 0>(hacc[j + jj][0], hacc[j + jj][1], a0, a1, a2, a3, b[2*jj + 0], b[2*jj + 1]);
                        } else {
                            mmms_mma_f32(acc[j + jj], a0, a1, a2, a3, b[2*jj + 0], b[2*jj + 1]);
                        }
                    }
                }
            });

            if constexpr (acc_f16) {
#pragma unroll
                for (int j = 0; j < ntn; ++j) {
                    const float2 lo = __half22float2(*(const half2 *) &hacc[j][0]);
                    const float2 hi = __half22float2(*(const half2 *) &hacc[j][1]);
                    acc[j][0] += lo.x;
                    acc[j][1] += lo.y;
                    acc[j][2] += hi.x;
                    acc[j][3] += hi.y;
                }
            }
        }

        if constexpr (glu) {
            // the gate warps pass their sums through smem (the y stages, all warps are done with them), the up warps write
            float * dz = a.dst + i13*a.s3 + i12*a.s2;
            float * xs = (float *) tile_y;
            constexpr int nh = ntn/2; // n tiles per exchange
            static_assert((nwarps/2)*nh*4*WARP_SIZE*sizeof(float) <= MMMS_NSTAGES*sbytes, "glu exchange does not fit");
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                __syncthreads();
                if (warp < nwarps/2) {
#pragma unroll
                    for (int jj = 0; jj < nh; ++jj) {
#pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            xs[((wr*nh + jj)*4 + l)*WARP_SIZE + lane] = acc[h*nh + jj][l];
                        }
                    }
                }
                __syncthreads();
                if (warp >= nwarps/2) {
#pragma unroll
                    for (int jj = 0; jj < nh; ++jj) {
#pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            const int m = tm*bm + mmms_out_row(wr, lane, l);
                            const int n = col0 + mmms_out_col(lane, h*nh + jj, l);
                            if (m < a.ne01 && n < a.ne1) {
                                const float is = a.inv_scale[(int64_t) iz*a.ne11_pad + n];
                                const float gv = xs[((wr*nh + jj)*4 + l)*WARP_SIZE + lane]*is;
                                dz[n*a.s1 + m] = ggml_cuda_op_glu_single((ggml_glu_op) a.glu_op, gv, acc[h*nh + jj][l]*is, a.glu_limit);
                            }
                        }
                    }
                }
            }
        } else if (ids) {
#pragma unroll
            for (int j = 0; j < ntn; ++j) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int m = tm*bm + mmms_out_row(warp, lane, l);
                    const int n = mmms_out_col(lane, j, l);
                    if (m < a.ne01 && n < ncols) {
                        const int pair = ids_pair[n];
                        a.dst[(pair / a.n_used)*a.s2 + (pair % a.n_used)*a.s1 + m] = acc[j][l]*a.inv_scale[ids_ycol[n]];
                    }
                }
            }
        } else if (kc0 == 0 && kc1 == nchunks) {
            float * dz = a.dst + i13*a.s3 + i12*a.s2;
#pragma unroll
            for (int j = 0; j < ntn; ++j) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int m = tm*bm + mmms_out_row(warp, lane, l);
                    const int n = col0 + mmms_out_col(lane, j, l);
                    if (m < a.ne01 && n < a.ne1) {
                        dz[n*a.s1 + m] = acc[j][l]*a.inv_scale[(int64_t) iz*a.ne11_pad + n];
                    }
                }
            }
        } else {
            // part of a tile, in the order of the thread values (coalesced)
            float * p = a.part + ((int64_t) 2*blockIdx.x + (seg == 0 ? 0 : 1))*bm*bn;
#pragma unroll
            for (int j = 0; j < ntn; ++j) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    p[((warp*ntn + j)*4 + l)*WARP_SIZE + lane] = acc[j][l];
                }
            }
        }

        it += kc1 - kc0;
        seg++;
        __syncthreads(); // the next tile reuses the y stages
    }
#else
    GGML_UNUSED_VARS(a, xw, s01, s02, s03);
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
}

// k_up: the type of the up weights with glu (the gate and up warps run the body of their own type, the barriers match)
template <int k, int k_up, int nwarps, int bn, bool acc_f16, bool glu = false, bool ids = false>
static __global__ void __launch_bounds__(nwarps*WARP_SIZE)
mul_mat_ms_c(const mmms_args a) {
    if constexpr (glu && k != k_up) {
        if (threadIdx.y >= nwarps/2) {
            mul_mat_ms_c_body<k_up, nwarps, bn, acc_f16, glu, false>(a, a.x_up, a.s01_up, a.s02_up, a.s03_up);
        } else {
            mul_mat_ms_c_body<k,    nwarps, bn, acc_f16, glu, false>(a, a.x,    a.s01,    a.s02,    a.s03);
        }
    } else if constexpr (glu) {
        const bool up = threadIdx.y >= nwarps/2;
        mul_mat_ms_c_body<k, nwarps, bn, acc_f16, glu, false>(a, up ? a.x_up : a.x, up ? a.s01_up : a.s01, up ? a.s02_up : a.s02, up ? a.s03_up : a.s03);
    } else {
        mul_mat_ms_c_body<k, nwarps, bn, acc_f16, glu, ids>(a, a.x, a.s01, a.s02, a.s03);
    }
}

// sums the parts of the tiles that more than one block worked on, in block order
template <int nwarps, int bn>
static __global__ void __launch_bounds__(nwarps*WARP_SIZE)
mul_mat_ms_c_fixup(const mmms_args a) {
    constexpr int bm  = 16*nwarps;
    constexpr int ntn = bn/8;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;

    const int     tile    = blockIdx.x;
    const int     nchunks = a.ne00/MMMS_BK;
    const int64_t total   = (int64_t) a.ntiles*nchunks;
    const int64_t it0     = (int64_t) tile*nchunks;
    const int64_t it1     = it0 + nchunks;

    // the block that has the first iteration of the tile
    int b = min((int64_t) a.nblocks - 1, it0*a.nblocks/total);
    while (b > 0 && mmms_it_begin(b, total, a.nblocks) > it0) {
        b--;
    }
    while (b + 1 < a.nblocks && mmms_it_begin(b + 1, total, a.nblocks) <= it0) {
        b++;
    }
    if (mmms_it_begin(b, total, a.nblocks) <= it0 && mmms_it_begin(b + 1, total, a.nblocks) >= it1) {
        return; // done by one block
    }

    float acc[ntn][4] = {};
    for (; b < a.nblocks; ++b) {
        const int64_t bb = mmms_it_begin(b, total, a.nblocks);
        const int64_t be = mmms_it_begin(b + 1, total, a.nblocks);
        if (bb >= it1) {
            break;
        }
        if (be <= bb) {
            continue; // no iterations
        }
        const int slot = bb / nchunks == tile ? 0 : 1; // first or last tile of block b
        const float * p = a.part + ((int64_t) 2*b + slot)*bm*bn;
#pragma unroll
        for (int j = 0; j < ntn; ++j) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                acc[j][l] += p[((warp*ntn + j)*4 + l)*WARP_SIZE + lane];
            }
        }
    }

    const int tn  = tile % a.ntiles_n;
    const int tm  = (tile / a.ntiles_n) % a.ntiles_m;
    const int iz  = tile / (a.ntiles_n*a.ntiles_m);
    const int i12 = iz % a.ne12;
    const int i13 = iz / a.ne12;
    float * dz = a.dst + i13*a.s3 + i12*a.s2;
#pragma unroll
    for (int j = 0; j < ntn; ++j) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int m = tm*bm + mmms_out_row(warp, lane, l);
            const int n = tn*bn + mmms_out_col(lane, j, l);
            if (m < a.ne01 && n < a.ne1) {
                dz[n*a.s1 + m] = acc[j][l]*a.inv_scale[(int64_t) iz*a.ne11_pad + n];
            }
        }
    }
}

// Small batches (MMMS_SMALL_MIN_COLS .. MMMS_SMALL_MAX_COLS columns, below that the exact mat-vec is cheaper): one mma tile
// holds all columns, so the weights are decoded once and the column count costs nearly nothing.
// Same values as above: w = f16(v*d), y = f16(y*scale), f32 sums.
// The weights are the B operand (n8 = 8 rows per block), the columns the A operand (rows 8..15 of A are zero), so a block
// streams 8 rows; its warps split them into super-chunks of MMMS_SC k values (warp w: super-chunks w, w + nwarps, ...).
// In a super-chunk thread t decodes the MMMS_SC/4 consecutive states MMMS_SC/4*t .. of row g = lane/4, so the 4 threads of
// a row read contiguous bytes, as runs u of 16 states that feed 4 mma steps each (step s: states 4*s + 0..3 of the run, as
// in mmms_decode_step). y is f16 with k = MMMS_SC/4*t + 16*u + i of a super-chunk stored at 64*u + 16*t + i: the B fragments
// of run u are 16 consecutive values and the 4 threads of a column read adjacent ones.
// The weight loads fetch 256 bytes into L2 (the other threads of the row and the next warps read the rest) and are evicted
// from L1 first (y stays). The words of the next super-chunk of a warp are loaded before the current one is decoded.
// When the row tiles fill few waves of the GPU the blocks (tile, part) also split K, the last block of a tile adds the
// parts in part order (deterministic).
#define MMMS_SMALL_MIN_COLS 4
#define MMMS_SMALL_MAX_COLS 8
#define MMMS_SMALL_NWARPS   8
#define MMMS_SC             256  // k values per super-chunk
#define MMMS_SMALL_CONVERT_SLICE 2048 // k values per block of mmms_small_convert_y

// y as f16 [ne12*ne13][ne11][ne10] with the column scale of mmms_convert_y, in the order above; block (col, part) converts
// k values part*MMMS_SMALL_CONVERT_SLICE .. of col and reads the whole column for the scale.
template <bool aligned>
static __global__ void __launch_bounds__(MMMS_CONVERT_NT)
mmms_small_convert_y(
        const float * __restrict__ x, half2 * __restrict__ y, float * __restrict__ inv_scale,
        const int ne10, const int ne11, const int ne12, const int64_t s11, const int64_t s12, const int64_t s13) {
    const int64_t col = blockIdx.x;
    const int     i11 = col % ne11;
    const int     i12 = (col / ne11) % ne12;
    const int64_t i13 = (col / ne11) / ne12;
    const int     tid = threadIdx.x;

    const float * xc = x + i13*s13 + i12*s12 + i11*s11;

    float amax = 0.0f;
    for (int i = 2*tid; i < ne10; i += 2*MMMS_CONVERT_NT) {
        const float2 v = aligned ? *(const float2 *) (xc + i) : make_float2(xc[i], xc[i + 1]);
        amax = fmaxf(amax, fmaxf(fabsf(v.x), fabsf(v.y)));
    }
    amax = warp_reduce_max(amax);
    __shared__ float amax_warp[MMMS_CONVERT_NT/WARP_SIZE];
    if (tid % WARP_SIZE == 0) {
        amax_warp[tid / WARP_SIZE] = amax;
    }
    __syncthreads();
    amax = 0.0f;
#pragma unroll
    for (int w = 0; w < MMMS_CONVERT_NT/WARP_SIZE; ++w) {
        amax = fmaxf(amax, amax_warp[w]);
    }

    // the scale of mmms_convert_y
    int e = 0;
    if (amax > 0.0f && isfinite(amax)) {
        frexpf(amax, &e);
        e = min(max(6 - e, -100), 100);
    } else {
        e = 0;
    }
    const float scale = ldexpf(1.0f, e/2)*ldexpf(1.0f, e - e/2);
    if (blockIdx.y == 0 && tid == 0) {
        inv_scale[col] = 1.0f/scale;
    }

    half2 * yc = y + col*(ne10/2);
    const int i1 = min(ne10, (int) (blockIdx.y + 1)*MMMS_SMALL_CONVERT_SLICE);
    for (int q = blockIdx.y*MMMS_SMALL_CONVERT_SLICE + 2*tid; q < i1; q += 2*MMMS_CONVERT_NT) {
        const int r = q % MMMS_SC;
        const int i = q - r + MMMS_SC/4*((r/16) % 4) + 16*(r/64) + r % 16;
        const float2 v = aligned ? *(const float2 *) (xc + i) : make_float2(xc[i], xc[i + 1]);
        yc[q/2] = __floats2half2_rn(v.x*scale, v.y*scale);
    }
}

// the words of the MMMS_SC/4 states of thread t in super-chunk h of a block start at bit 18 of stream word
// k*MMMS_SC/128*t + k*MMMS_SC/32*h of the block
template <int k>
struct mmms_sc_block {
    static constexpr int nw = (MMMS_SC/4*k + 31)/32 + 1;
};

template <int k>
static __device__ __forceinline__ void mmms_sc_load(
        const char * __restrict__ xr, const int ib, const int base, uint32_t (&w)[mmms_sc_block<k>::nw], half2 & d) {
    using bl = mmms_block<k>;
    const uint32_t * wp = (const uint32_t *) (xr + (int64_t) ib*bl::bs) + base;
#pragma unroll
    for (int i = 0; i < mmms_sc_block<k>::nw; ++i) {
#if defined(TURING_MMA_AVAILABLE)
        asm volatile("ld.global.L1::evict_first.L2::256B.u32 %0, [%1];" : "=r"(w[i]) : "l"(wp + i));
#else
        w[i] = wp[i];
#endif // defined(TURING_MMA_AVAILABLE)
    }
    d = mmms_load_d(xr, ib, bl::bs);
}

// code bits of run u (states 16*u .. 16*u + 15 of the thread), MSB first from its first state; first: the words start at word 0
template <int k, int u>
static __device__ __forceinline__ void mmms_sc_prep(
        const uint32_t (&w)[mmms_sc_block<k>::nw], const bool first, uint32_t (&wn)[mmms_block<k>::nwn]) {
    constexpr int nw  = mmms_sc_block<k>::nw;
    constexpr int nwn = mmms_block<k>::nwn;
    constexpr int off = 32 - GGML_MS_L + 16*k*u;
    constexpr int o   = off / 32;
    constexpr int sh  = off % 32;
    uint32_t wb[nwn + 1];
#pragma unroll
    for (int i = 0; i <= nwn; ++i) {
        const int j = o + i < nw ? o + i : nw - 1; // a clamped word only fills bits after the last state
        wb[i] = __byte_perm(w[j], 0, 0x0123);
        if (j == 0) {
            wb[i] = first ? w[0] >> 16 : wb[i]; // word 0 of the block holds d (low) and s0 (high)
        }
    }
#pragma unroll
    for (int i = 0; i < nwn; ++i) {
        wn[i] = sh == 0 ? wb[i] : __funnelshift_l(wb[i + 1], wb[i], sh);
    }
}

// thread (g, t): row g of the tile (B operand), column g (A operand); its sums are column g, rows 2*t and 2*t + 1.
// grid: (row tiles * ksplit, 1, ne12*ne13), ksplit in a.nblocks
template <int k>
static __global__ void __launch_bounds__(MMMS_SMALL_NWARPS*WARP_SIZE)
mul_mat_ms_c_small(const mmms_args a, int * __restrict__ counters) {
#if defined(TURING_MMA_AVAILABLE)
    using bl  = mmms_block<k>;
    using sbl = mmms_sc_block<k>;
    constexpr int nwarps = MMMS_SMALL_NWARPS;
    constexpr int scpb   = QK_MS/MMMS_SC; // super-chunks per block
    constexpr int nruns  = MMMS_SC/64;    // runs of 16 states per thread and super-chunk

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int g    = lane / 4;
    const int t    = lane % 4;

    // the parts of a tile are adjacent blocks
    const int ksplit = a.nblocks;
    const int ks     = blockIdx.x % ksplit;
    const int tm     = blockIdx.x / ksplit;
    const int iz     = blockIdx.z;
    const int i12    = iz % a.ne12;
    const int i13    = iz / a.ne12;
    const int row0   = 8*tm;

    const char * xr = a.x + (i13/a.r3)*a.s03 + (i12/a.r2)*a.s02 + min(row0 + g, a.ne01 - 1)*a.s01;

    // column g of y (the threads of the columns >= ne1 use zeros)
    const bool    has_col = g < a.ne1;
    const uint4 * yc      = (const uint4 *) (a.y + ((int64_t) iz*a.ne1 + (has_col ? g : 0))*(a.ne00/2));

    const int nsc = a.ne00/MMMS_SC;
    const int sc0 = (int) ((int64_t) ks*nsc/ksplit);
    const int sc1 = (int) ((int64_t) (ks + 1)*nsc/ksplit);

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint32_t wp[sbl::nw];
    half2    dp;
    auto load_sc = [&](const int isc) {
        mmms_sc_load<k>(xr, isc / scpb, k*MMMS_SC/128*t + k*MMMS_SC/32*(isc % scpb), wp, dp);
    };
    if (sc0 + warp < sc1) {
        load_sc(sc0 + warp);
    }

    for (int isc = sc0 + warp; isc < sc1; isc += nwarps) {
        const bool first = t == 0 && isc % scpb == 0;

        // y of all runs of the super-chunk, then the words of the next super-chunk of the warp
        uint4 y[nruns][2];
#pragma unroll
        for (int u = 0; u < nruns; ++u) {
            y[u][0] = make_uint4(0, 0, 0, 0);
            y[u][1] = make_uint4(0, 0, 0, 0);
            if (has_col) {
                const int i8 = (isc*MMMS_SC + 64*u + 16*t)/8;
                y[u][0] = yc[i8 + 0];
                y[u][1] = yc[i8 + 1];
            }
        }

        uint32_t w[sbl::nw];
#pragma unroll
        for (int i = 0; i < sbl::nw; ++i) {
            w[i] = wp[i];
        }
        const half2 d = dp;
        if (isc + nwarps < sc1) {
            load_sc(isc + nwarps);
        }

        static_for<0, nruns>([&](auto u_) {
            constexpr int u = decltype(u_)::value;
            const uint32_t b[8] = {y[u][0].x, y[u][0].y, y[u][0].z, y[u][0].w, y[u][1].x, y[u][1].y, y[u][1].z, y[u][1].w};

            uint32_t wn[bl::nwn];
            mmms_sc_prep<k, u>(w, first, wn);
            static_for<0, 4>([&](auto s_) {
                constexpr int s = decltype(s_)::value;
                uint32_t f0, f1;
                mmms_decode_step<k, s>(wn, d, f0, f1);
                mmms_mma_f32(acc, b[2*s + 0], 0, b[2*s + 1], 0, f0, f1);
            });
        });
    }

    // sums of the warps in warp order
    __shared__ float red[nwarps - 1][2][WARP_SIZE];
    if (warp > 0) {
        red[warp - 1][0][lane] = acc[0];
        red[warp - 1][1][lane] = acc[1];
    }
    __syncthreads();
    if (warp > 0) {
        return;
    }
#pragma unroll
    for (int i = 0; i < nwarps - 1; ++i) {
        acc[0] += red[i][0][lane];
        acc[1] += red[i][1][lane];
    }

    if (ksplit > 1) {
        // part ks of the tile, then the last block of the tile adds the parts in order
        const int64_t tile = (int64_t) iz*(gridDim.x/ksplit) + tm;
        float * part = a.part + tile*ksplit*(2*WARP_SIZE);
        part[(2*ks + 0)*WARP_SIZE + lane] = acc[0];
        part[(2*ks + 1)*WARP_SIZE + lane] = acc[1];
        __threadfence();
        int last = 0;
        if (lane == 0) {
            last = atomicAdd(counters + tile, 1) == ksplit - 1;
        }
        if (!__shfl_sync(0xFFFFFFFF, last, 0)) {
            return;
        }
        __threadfence();
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        for (int i = 0; i < ksplit; ++i) {
            sum0 += __ldcg(part + (2*i + 0)*WARP_SIZE + lane);
            sum1 += __ldcg(part + (2*i + 1)*WARP_SIZE + lane);
        }
        acc[0] = sum0;
        acc[1] = sum1;
    }

    if (has_col) {
        float * dz = a.dst + i13*a.s3 + i12*a.s2 + g*a.s1;
        const float is = a.inv_scale[(int64_t) iz*a.ne1 + g];
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            const int row = row0 + 2*t + l;
            if (row < a.ne01) {
                dz[row] = acc[l]*is;
            }
        }
    }
#else
    GGML_UNUSED_VARS(a, counters);
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
}

// y_conv, inv_scale_conv: where src1 is converted to (a.y and a.inv_scale, writable), nullptr if a.y already holds it
template <int k>
static void mul_mat_ms_c_small_run(ggml_backend_cuda_context & ctx, mmms_args a, const ggml_tensor * src1, half2 * y_conv, float * inv_scale_conv) {
    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb);
    cudaStream_t stream = ctx.stream();

    const int id = ggml_cuda_get_device();
    static int occupancy[GGML_CUDA_MAX_DEVICES] = {0};
    if (occupancy[id] == 0) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy[id], mul_mat_ms_c_small<k>, MMMS_SMALL_NWARPS*WARP_SIZE, 0));
        occupancy[id] = std::max(occupancy[id], 1);
    }

    // K split over blocks only when the row tiles fill less than a few waves: whole rows per block stream best
    const int ntiles_m = (a.ne01 + 7)/8;
    const int ntiles   = ntiles_m*a.ne12*a.ne13;
    const int nslots   = ggml_cuda_info().devices[id].nsm*occupancy[id];
    const int nsc      = a.ne00/MMMS_SC;
    const int ksplit   = std::max(1, std::min((4*nslots + ntiles - 1)/ntiles, nsc/MMMS_SMALL_NWARPS));

    ggml_cuda_pool_alloc<int>   counters(ctx.pool(), ksplit > 1 ? ntiles : 1);
    ggml_cuda_pool_alloc<float> part(ctx.pool(), ksplit > 1 ? (size_t) ntiles*ksplit*2*WARP_SIZE : 1);
    a.part    = part.get();
    a.nblocks = ksplit;

    if (ksplit > 1) {
        CUDA_CHECK(cudaMemsetAsync(counters.get(), 0, ntiles*sizeof(int), stream));
    }

    if (y_conv != nullptr) {
        const bool aligned = (uintptr_t) src1->data % 8 == 0 && nb11 % 8 == 0 && nb12 % 8 == 0 && nb13 % 8 == 0;
        const dim3 grid(ne12*ne13*ne11, (ne10 + MMMS_SMALL_CONVERT_SLICE - 1)/MMMS_SMALL_CONVERT_SLICE);
        if (aligned) {
            mmms_small_convert_y<true><<<grid, MMMS_CONVERT_NT, 0, stream>>>((const float *) src1->data, y_conv,
                inv_scale_conv, ne10, ne11, ne12, nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float));
        } else {
            mmms_small_convert_y<false><<<grid, MMMS_CONVERT_NT, 0, stream>>>((const float *) src1->data, y_conv,
                inv_scale_conv, ne10, ne11, ne12, nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 grid(ntiles_m*ksplit, 1, a.ne12*a.ne13);
    const dim3 block(WARP_SIZE, MMMS_SMALL_NWARPS, 1);
    mul_mat_ms_c_small<k><<<grid, block, 0, stream>>>(a, counters.get());
}

static void ggml_cuda_mul_mat_ms_small(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const bool cache_ok) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    GGML_ASSERT(ne00 % MMMS_SC == 0);

    // the f16 y is kept for the next matmuls of this graph evaluation that read src1 (Q, K, V; gate, up). Only for the
    // MUL_MAT nodes themselves (cache_ok: callers such as MUL_MAT_ID hold pool memory of their own) and with one stream
    // (the pools are per stream).
    const size_t y_size    = (size_t) ne12*ne13*ne11*ne10*sizeof(half);
    const size_t y_alloc   = y_size + (size_t) ne12*ne13*ne11*sizeof(float);
    const bool   use_cache = cache_ok && ctx.stream_context().concurrent_events.empty();
    const bool   cached    = use_cache && ctx.mmms_y_src == src1 && !ctx.mmms_y_ids && ctx.mmms_y;
    ggml_cuda_pool_alloc<char> y_tmp(ctx.pool());
    char * yb;
    if (cached) {
        yb = ctx.mmms_y->get();
    } else if (use_cache) {
        ctx.mmms_y.reset();
        ctx.mmms_y     = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), y_alloc);
        ctx.mmms_y_src = src1;
        ctx.mmms_y_ids = false;
        yb = ctx.mmms_y->get();
    } else {
        yb = y_tmp.alloc(y_alloc);
    }

    mmms_args a = {};
    a.x = (const char *) src0->data; a.y = (const half2 *) yb; a.inv_scale = (const float *) (yb + y_size);
    a.dst = (float *) dst->data; a.part = nullptr;
    a.ne00 = ne00; a.ne01 = ne01; a.s01 = nb01; a.s02 = nb02; a.s03 = nb03;
    a.ne1 = ne11; a.ne11_pad = ne11; a.ne12 = ne12; a.ne13 = ne13; a.r2 = ne12/ne02; a.r3 = ne13/ne03;
    a.s1 = nb1/sizeof(float); a.s2 = nb2/sizeof(float); a.s3 = nb3/sizeof(float);

    half2 * y_conv         = cached ? nullptr : (half2 *) yb;
    float * inv_scale_conv = cached ? nullptr : (float *) (yb + y_size);

    switch (src0->type) {
        case GGML_TYPE_MS_C1: mul_mat_ms_c_small_run<1>(ctx, a, src1, y_conv, inv_scale_conv); break;
        case GGML_TYPE_MS_C2: mul_mat_ms_c_small_run<2>(ctx, a, src1, y_conv, inv_scale_conv); break;
        case GGML_TYPE_MS_C3: mul_mat_ms_c_small_run<3>(ctx, a, src1, y_conv, inv_scale_conv); break;
        case GGML_TYPE_MS_C4: mul_mat_ms_c_small_run<4>(ctx, a, src1, y_conv, inv_scale_conv); break;
        case GGML_TYPE_MS_C5: mul_mat_ms_c_small_run<5>(ctx, a, src1, y_conv, inv_scale_conv); break;
        case GGML_TYPE_MS_C6: mul_mat_ms_c_small_run<6>(ctx, a, src1, y_conv, inv_scale_conv); break;
        default:
            GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_should_use_mmms(enum ggml_type type, int cc, int64_t ne11) {
    switch (type) {
        case GGML_TYPE_MS_C1:
        case GGML_TYPE_MS_C2:
        case GGML_TYPE_MS_C3:
        case GGML_TYPE_MS_C4:
        case GGML_TYPE_MS_C5:
        case GGML_TYPE_MS_C6:
            break;
        default:
            return false;
    }
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS
    static const bool disabled = getenv("GGML_CUDA_DISABLE_MMMS") != nullptr;
    const bool small = ne11 >= MMMS_SMALL_MIN_COLS && ne11 <= MMMS_SMALL_MAX_COLS;
    return !disabled && GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && (ne11 > MMVQ_MAX_BATCH_SIZE || small);
}

template <int k, int nwarps, int bn, bool acc_f16, bool glu = false, int k_up = k, bool ids = false>
static void mul_mat_ms_c_launch(ggml_backend_cuda_context & ctx, mmms_args a, const bool stream_k) {
    cudaStream_t stream = ctx.stream();
    // ids: the columns of the tile after the y stages
    constexpr size_t nbytes_shared = mmms_nbytes_shared<bn>() + (ids ? 2*bn*sizeof(int) : 0);
    auto kernel = mul_mat_ms_c<k, k_up, nwarps, bn, acc_f16, glu, ids>;
    CUDA_SET_SHARED_MEMORY_LIMIT(kernel, nbytes_shared);

    const int id = ggml_cuda_get_device();
    static int occupancy[GGML_CUDA_MAX_DEVICES] = {0};
    if (occupancy[id] == 0) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy[id], kernel, nwarps*WARP_SIZE, nbytes_shared));
        occupancy[id] = std::max(occupancy[id], 1);
    }
    const int nslots = ggml_cuda_info().devices[id].nsm*occupancy[id];

    // stream-k when the last wave of tiles leaves blocks idle, unless there are fewer tiles than blocks and few chunks
    // of a wide tile: then nearly every tile is split and adding the parts costs more than the idle blocks.
    // Small n tiles (small batches) are always split, the parts are small and the weights are read once either way.
    const int    nchunks = a.ne00/MMMS_BK;
    const double eff     = (double) a.ntiles/((a.ntiles + nslots - 1)/nslots*nslots);
    const bool   use_sk  = stream_k && eff < 0.98 && (a.ntiles >= nslots || nchunks >= 64 || bn <= 64 || 2*a.ntiles <= nslots);

    const dim3 block(WARP_SIZE, nwarps, 1);
    if (!use_sk) {
        // one tile per block
        a.nblocks = a.ntiles;
        kernel<<<a.ntiles, block, nbytes_shared, stream>>>(a);
        return;
    }

    // at least MMMS_SK_MIN_CHUNKS chunks per block
    const int64_t total = (int64_t) a.ntiles*nchunks;
    a.nblocks = (int) std::max<int64_t>(1, std::min<int64_t>(nslots, total/MMMS_SK_MIN_CHUNKS));
    ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) 2*a.nblocks*16*nwarps*bn);
    a.part = part.get();
    kernel<<<a.nblocks, block, nbytes_shared, stream>>>(a);
    mul_mat_ms_c_fixup<nwarps, bn><<<a.ntiles, block, 0, stream>>>(a);
}

// gate/up fusion: the 128 column tile, nwarps/2 warps on the gate rows and as many on the same rows of the up weights
#define MMMS_GLU_BN 128

// nwarps 16: 128 output rows per tile, 8: 64 (the tile choice is in ggml_cuda_mul_mat_ms_impl)
template <int k, int k_up>
static void mul_mat_ms_c_glu(ggml_backend_cuda_context & ctx, const int nwarps, const bool acc_f16, mmms_args a) {
    a.ntiles_m = (a.ne01 + 8*nwarps - 1)/(8*nwarps);
    a.ntiles_n = a.ne11_pad/MMMS_GLU_BN;
    a.ntiles   = a.ntiles_m*a.ntiles_n*a.ne12*a.ne13;
    if (nwarps == 16) {
        if (acc_f16) { mul_mat_ms_c_launch<k, 16, MMMS_GLU_BN, true,  true, k_up>(ctx, a, false); }
        else         { mul_mat_ms_c_launch<k, 16, MMMS_GLU_BN, false, true, k_up>(ctx, a, false); }
    } else {
        if (acc_f16) { mul_mat_ms_c_launch<k,  8, MMMS_GLU_BN, true,  true, k_up>(ctx, a, false); }
        else         { mul_mat_ms_c_launch<k,  8, MMMS_GLU_BN, false, true, k_up>(ctx, a, false); }
    }
}

// gate/up types of a fused pair: up to MMMS_GLU_MAX_DK bits apart
#define MMMS_GLU_MAX_DK 2

static int mmms_type_bits(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_MS_C1: return 1;
        case GGML_TYPE_MS_C2: return 2;
        case GGML_TYPE_MS_C3: return 3;
        case GGML_TYPE_MS_C4: return 4;
        case GGML_TYPE_MS_C5: return 5;
        case GGML_TYPE_MS_C6: return 6;
        default:              return 0;
    }
}

template <int k>
static void mul_mat_ms_c_glu_switch(ggml_backend_cuda_context & ctx, const int nwarps, const bool acc_f16, const int k_up, const mmms_args & a) {
#define MMMS_GLU_CASE(ku_) \
    if constexpr (ku_ >= k - MMMS_GLU_MAX_DK && ku_ <= k + MMMS_GLU_MAX_DK) { \
        if (k_up == ku_) { \
            mul_mat_ms_c_glu<k, ku_>(ctx, nwarps, acc_f16, a); \
            return; \
        } \
    }
    MMMS_GLU_CASE(1) MMMS_GLU_CASE(2) MMMS_GLU_CASE(3) MMMS_GLU_CASE(4) MMMS_GLU_CASE(5) MMMS_GLU_CASE(6)
#undef MMMS_GLU_CASE
    GGML_ABORT("fatal error");
}


template <int k>
static void mul_mat_ms_c_switch(ggml_backend_cuda_context & ctx, const int nwarps, const int bn, const bool acc_f16, const bool stream_k, mmms_args a) {
#define MMMS_CASE(nw_, bn_)                                                                   \
    if (nwarps == nw_ && bn == bn_) {                                                         \
        a.ntiles_m = (a.ne01 + 16*nw_ - 1)/(16*nw_);                                          \
        a.ntiles_n = a.ne11_pad/bn_;                                                          \
        a.ntiles   = a.ntiles_m*a.ntiles_n*a.ne12*a.ne13;                                     \
        if (acc_f16) { mul_mat_ms_c_launch<k, nw_, bn_, true >(ctx, a, stream_k); }           \
        else         { mul_mat_ms_c_launch<k, nw_, bn_, false>(ctx, a, stream_k); }           \
        return;                                                                               \
    }
    MMMS_CASE(16, 128) MMMS_CASE(8, 128) MMMS_CASE(8, 64) MMMS_CASE(4, 32) MMMS_CASE(4, 16)
#undef MMMS_CASE
    GGML_ABORT("fatal error");
}

// f16 sums per chunk (added to f32 sums) or f32 sums. The f32 sums are exact up to the f16 inputs (KLD ~1e-6 vs ~4e-6),
// GeForce and TITAN cards run them at half the tensor core rate, data center and workstation cards at full rate.
// The GB10 (DGX Spark, RTX Spark) measured as fast with f32 sums, so it keeps them. GGML_CUDA_MMMS_ACC=f16|f32 overrides the choice.
static bool mmms_use_f16_sums(const int id) {
    static int choice[GGML_CUDA_MAX_DEVICES] = {0}; // 0: not decided yet, 1: f16 sums, 2: f32 sums
    if (choice[id] == 0) {
        const char * e = getenv("GGML_CUDA_MMMS_ACC");
        if (e != nullptr && strcmp(e, "f32") == 0) {
            choice[id] = 2;
        } else if (e != nullptr && strcmp(e, "f16") == 0) {
            choice[id] = 1;
        } else {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
            const bool half_rate = strstr(prop.name, "GeForce") != nullptr || strstr(prop.name, "TITAN") != nullptr;
            choice[id] = half_rate ? 1 : 2;
        }
    }
    return choice[id] == 1;
}

// src1 as the f16 y of the tensor core kernel: column (i13*ne12 + i12)*ne11_pad + i11, zero columns from ne11 to ne11_pad
static void mmms_convert_y_cuda(const ggml_tensor * src1, half2 * y, float * inv_scale, const int64_t ne11_pad, cudaStream_t stream) {
    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb);

    const int r1 = (ne10/8 + MMMS_CONVERT1_NT - 1)/MMMS_CONVERT1_NT;
    const bool aligned = (uintptr_t) src1->data % 8 == 0 && nb11 % 8 == 0 && nb12 % 8 == 0 && nb13 % 8 == 0;
    const dim3 grid(ne12*ne13*ne11_pad);
#define MMMS_CONVERT1_CASE(r_) \
    if (r1 == r_) { \
        if (aligned) { mmms_convert_y1<true,  r_><<<grid, MMMS_CONVERT1_NT, 0, stream>>>((const float *) src1->data, (uint4 *) y, inv_scale, \
            ne10, ne11, ne11_pad, ne12, nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float)); } \
        else         { mmms_convert_y1<false, r_><<<grid, MMMS_CONVERT1_NT, 0, stream>>>((const float *) src1->data, (uint4 *) y, inv_scale, \
            ne10, ne11, ne11_pad, ne12, nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float)); } \
    } else
    MMMS_CONVERT1_CASE(1) MMMS_CONVERT1_CASE(2) MMMS_CONVERT1_CASE(3) MMMS_CONVERT1_CASE(4) MMMS_CONVERT1_CASE(5) MMMS_CONVERT1_CASE(6)
#undef MMMS_CONVERT1_CASE
    if (aligned) {
        mmms_convert_y<true><<<grid, MMMS_CONVERT_NT, 0, stream>>>((const float *) src1->data, (uint4 *) y, inv_scale,
            ne10, ne11, ne11_pad, ne12, nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float));
    } else {
        mmms_convert_y<false><<<grid, MMMS_CONVERT_NT, 0, stream>>>((const float *) src1->data, (uint4 *) y, inv_scale,
            ne10, ne11, ne11_pad, ne12, nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float));
    }
    CUDA_CHECK(cudaGetLastError());
}

static void ggml_cuda_mul_mat_ms_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src0_up,
        const ggml_tensor * src1, ggml_tensor * dst, const bool cache_ok, const int glu_op, const float glu_limit) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == sizeof(float));
    GGML_ASSERT(nb0  == sizeof(float));
    GGML_ASSERT(ne00 % QK_MS == 0 && ne00 == ne10);
    GGML_ASSERT(ne12 % ne02 == 0 && ne13 % ne03 == 0);

    if (ne11 <= MMMS_SMALL_MAX_COLS) {
        GGML_ASSERT(src0_up == nullptr);
        ggml_cuda_mul_mat_ms_small(ctx, src0, src1, dst, cache_ok);
        return;
    }

    cudaStream_t stream = ctx.stream();

    // n tile: the smallest that holds the batch (small batches waste less of the mma on padding columns)
    const int bn     = ne11 <= 16 ? 16 : ne11 <= 32 ? 32 : ne11 <= 64 ? 64 : 128;
    GGML_ASSERT(src0_up == nullptr || bn == MMMS_GLU_BN);

    const int  id      = ggml_cuda_get_device();
    const bool acc_f16 = mmms_use_f16_sums(id);

    // the 128 column tile: 16 warps (256 rows, or 128 gate + 128 up rows) halve the y traffic per weight, which pays where
    // the f32 sums run at the full rate (the f16 sums need 96 accumulator registers, more than the 128 per thread of a
    // 512 thread block leave room for) and the tiles still fill the GPU (one block per SM); else 8 warps.
    // GGML_CUDA_MMMS_ROWS=128|256 forces the rows of the plain tile (with glu: 64|128 output rows).
    int nwarps = bn <= 32 ? 4 : 8;
    if (bn == 128) {
        static const int rows_env = getenv("GGML_CUDA_MMMS_ROWS") ? atoi(getenv("GGML_CUDA_MMMS_ROWS")) : 0;
        const int64_t ntiles16 = (ne01 + (src0_up ? 127 : 255))/(src0_up ? 128 : 256) * ((ne11 + 127)/128) * ne12*ne13;
        const bool    wide     = rows_env ? rows_env == 256 : !acc_f16 && ntiles16 >= ggml_cuda_info().devices[id].nsm;
        nwarps = wide ? 16 : 8;
    }

    const bool stream_k = true;

    const int ne11_pad = GGML_PAD(ne11, bn);

    // the f16 y is kept for the next matmuls of this graph evaluation that read src1, as in ggml_cuda_mul_mat_ms_small
    // (a src1 always takes the same path, so the layouts of the two paths never mix)
    const size_t y_size    = (size_t) ne12*ne13*ne11_pad*ne10*sizeof(half);
    const size_t y_alloc   = y_size + (size_t) ne12*ne13*ne11_pad*sizeof(float);
    const bool   use_cache = cache_ok && ctx.stream_context().concurrent_events.empty();
    const bool   cached    = use_cache && ctx.mmms_y_src == src1 && !ctx.mmms_y_ids && ctx.mmms_y;
    ggml_cuda_pool_alloc<char> y_tmp(ctx.pool());
    char * yb;
    if (cached) {
        yb = ctx.mmms_y->get();
    } else if (use_cache) {
        ctx.mmms_y.reset();
        ctx.mmms_y     = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), y_alloc);
        ctx.mmms_y_src = src1;
        ctx.mmms_y_ids = false;
        yb = ctx.mmms_y->get();
    } else {
        yb = y_tmp.alloc(y_alloc);
    }
    half2 * y         = (half2 *) yb;
    float * inv_scale = (float *) (yb + y_size);
    if (!cached) {
        mmms_convert_y_cuda(src1, y, inv_scale, ne11_pad, stream);
    }

    mmms_args a = {};
    a.x = (const char *) src0->data; a.y = y; a.inv_scale = inv_scale; a.dst = (float *) dst->data; a.part = nullptr;
    a.x_up = src0_up ? (const char *) src0_up->data : nullptr; a.glu_op = glu_op; a.glu_limit = glu_limit;
    a.s01_up = src0_up ? src0_up->nb[1] : 0; a.s02_up = src0_up ? src0_up->nb[2] : 0; a.s03_up = src0_up ? src0_up->nb[3] : 0;
    a.ne00 = ne00; a.ne01 = ne01; a.s01 = nb01; a.s02 = nb02; a.s03 = nb03;
    a.ne1 = ne11; a.ne11_pad = ne11_pad; a.ne12 = ne12; a.ne13 = ne13; a.r2 = ne12/ne02; a.r3 = ne13/ne03;
    a.s1 = nb1/sizeof(float); a.s2 = nb2/sizeof(float); a.s3 = nb3/sizeof(float);

    if (src0_up) {
        const int k_up = mmms_type_bits(src0_up->type);
        switch (src0->type) {
            case GGML_TYPE_MS_C1: mul_mat_ms_c_glu_switch<1>(ctx, nwarps, acc_f16, k_up, a); break;
            case GGML_TYPE_MS_C2: mul_mat_ms_c_glu_switch<2>(ctx, nwarps, acc_f16, k_up, a); break;
            case GGML_TYPE_MS_C3: mul_mat_ms_c_glu_switch<3>(ctx, nwarps, acc_f16, k_up, a); break;
            case GGML_TYPE_MS_C4: mul_mat_ms_c_glu_switch<4>(ctx, nwarps, acc_f16, k_up, a); break;
            case GGML_TYPE_MS_C5: mul_mat_ms_c_glu_switch<5>(ctx, nwarps, acc_f16, k_up, a); break;
            case GGML_TYPE_MS_C6: mul_mat_ms_c_glu_switch<6>(ctx, nwarps, acc_f16, k_up, a); break;
            default:
                GGML_ABORT("fatal error");
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    switch (src0->type) {
        case GGML_TYPE_MS_C1: mul_mat_ms_c_switch<1>(ctx, nwarps, bn, acc_f16, stream_k, a); break;
        case GGML_TYPE_MS_C2: mul_mat_ms_c_switch<2>(ctx, nwarps, bn, acc_f16, stream_k, a); break;
        case GGML_TYPE_MS_C3: mul_mat_ms_c_switch<3>(ctx, nwarps, bn, acc_f16, stream_k, a); break;
        case GGML_TYPE_MS_C4: mul_mat_ms_c_switch<4>(ctx, nwarps, bn, acc_f16, stream_k, a); break;
        case GGML_TYPE_MS_C5: mul_mat_ms_c_switch<5>(ctx, nwarps, bn, acc_f16, stream_k, a); break;
        case GGML_TYPE_MS_C6: mul_mat_ms_c_switch<6>(ctx, nwarps, bn, acc_f16, stream_k, a); break;
        default:
            GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mul_mat_ms(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // only a MUL_MAT node of the graph may keep its f16 y: the MUL_MAT_ID fallback calls this with slices that also have
    // op MUL_MAT (for their precision), but no sources, and it frees pool memory of its own after them
    const bool cache_ok = dst->op == GGML_OP_MUL_MAT && dst->src[0] == src0 && dst->src[1] == src1;
    ggml_cuda_mul_mat_ms_impl(ctx, src0, nullptr, src1, dst, cache_ok, -1, 0.0f);
}

bool ggml_cuda_should_use_mmms_id(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst, const int cc) {
    if (!ggml_cuda_should_use_mmms(src0->type, cc, MMVQ_MAX_BATCH_SIZE + 1)) {
        return false;
    }
    // one sample; rows of contiguous values; per token one src1 column, or one for each used expert; the limits of the ids helper
    return src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 &&
        src0->ne[3] == 1 && src1->ne[3] == 1 && src0->ne[0] % QK_MS == 0 &&
        src0->nb[0] == ggml_type_size(src0->type) && src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) &&
        ids->nb[0] == sizeof(int32_t) && (src1->ne[1] == 1 || src1->ne[1] == ids->ne[0]) &&
        ids->ne[0] < (1 << 10) && src1->ne[2] < (1 << 22);
}

template <int k>
static void mul_mat_ms_c_id_switch(ggml_backend_cuda_context & ctx, const int nwarps, const int bn, const bool acc_f16, const mmms_args & a) {
#define MMMS_ID_CASE(nw_, bn_)                                                                              \
    if (nwarps == nw_ && bn == bn_) {                                                                       \
        if (acc_f16) { mul_mat_ms_c_launch<k, nw_, bn_, true,  false, k, true>(ctx, a, /*stream_k =*/ false); } \
        else         { mul_mat_ms_c_launch<k, nw_, bn_, false, false, k, true>(ctx, a, /*stream_k =*/ false); } \
        return;                                                                                             \
    }
    MMMS_ID_CASE(8, 128) MMMS_ID_CASE(8, 64) MMMS_ID_CASE(4, 32) MMMS_ID_CASE(4, 16)
#undef MMMS_ID_CASE
    GGML_ABORT("fatal error");
}

void ggml_cuda_mul_mat_ms_id(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32);
    GGML_ASSERT(nb00 == ggml_type_size(src0->type) && nb10 == sizeof(float) && nb0 == sizeof(float) && ids->nb[0] == sizeof(int32_t));
    GGML_ASSERT(ne00 % QK_MS == 0 && ne00 == ne10);
    GGML_ASSERT(ne03 == 1 && ne13 == 1);

    cudaStream_t stream = ctx.stream();

    const int n_used  = ids->ne[0];
    const int n_pairs = ne12*n_used;
    GGML_ASSERT(ne1 == n_used && ne2 == ne12 && (ne11 == 1 || ne11 == n_used));

    // column tile: the mean number of (token, slot) pairs per expert, rounded up to a tile width; GGML_CUDA_MMMS_ID_BN overrides
    static const int bn_env = getenv("GGML_CUDA_MMMS_ID_BN") ? atoi(getenv("GGML_CUDA_MMMS_ID_BN")) : 0;
    const int64_t mean   = (n_pairs + ne02 - 1)/ne02;
    const int     bn     = bn_env == 16 || bn_env == 32 || bn_env == 64 || bn_env == 128 ? bn_env :
                           mean <= 16 ? 16 : mean <= 32 ? 32 : mean <= 64 ? 64 : 128;
    const int     nwarps = bn <= 32 ? 4 : 8;
    const bool    acc_f16 = mmms_use_f16_sums(ggml_cuda_get_device());

    // y: the src1 columns in their own order (a pair reads the column of its token and slot), kept for the next MUL_MAT_ID of
    // this graph evaluation that reads src1 (gate and up); allocated before the index buffers, which are freed first
    const int64_t ncols_y   = ne11*ne12;
    const size_t  y_size    = (size_t) ncols_y*ne10*sizeof(half);
    const size_t  y_alloc   = y_size + (size_t) ncols_y*sizeof(float);
    const bool    use_cache = ctx.stream_context().concurrent_events.empty();
    const bool    cached    = use_cache && ctx.mmms_y_src == src1 && ctx.mmms_y_ids && ctx.mmms_y;
    ggml_cuda_pool_alloc<char> y_tmp(ctx.pool());
    char * yb;
    if (cached) {
        yb = ctx.mmms_y->get();
    } else if (use_cache) {
        ctx.mmms_y.reset();
        ctx.mmms_y     = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), y_alloc);
        ctx.mmms_y_src = src1;
        ctx.mmms_y_ids = true;
        yb = ctx.mmms_y->get();
    } else {
        yb = y_tmp.alloc(y_alloc);
    }
    half2 * y         = (half2 *) yb;
    float * inv_scale = (float *) (yb + y_size);
    if (!cached) {
        mmms_convert_y_cuda(src1, y, inv_scale, ne11, stream);
    }

    // the pairs sorted by expert, and the column tiles of each expert (empty tiles where negative ids leave seams unwritten)
    const int ntj = (n_pairs + bn - 1)/bn + ne02;
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_pairs);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_pairs);
    ggml_cuda_pool_alloc<ggml_cuda_mm_id_tile_spec> tile_spec(ctx.pool(), ntj);
    CUDA_CHECK(cudaMemsetAsync(tile_spec.get(), 0, ntj*sizeof(ggml_cuda_mm_id_tile_spec), stream));
    {
        const int si1  = ids->nb[1] / sizeof(int32_t);
        const int sis1 = nb12 / nb11;
        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), tile_spec.get(),
            ne02, ne12, n_used, ne11, bn, si1, sis1, /*write_inverse =*/ false, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    mmms_args a = {};
    a.x = (const char *) src0->data; a.y = y; a.inv_scale = inv_scale; a.dst = (float *) dst->data; a.part = nullptr;
    a.ne00 = ne00; a.ne01 = ne01; a.s01 = nb01; a.s02 = nb02; a.s03 = nb03;
    a.ne1 = ne1; a.ne11_pad = ne11; a.ne12 = 1; a.ne13 = 1; a.r2 = 1; a.r3 = 1;
    a.s1 = nb1/sizeof(float); a.s2 = nb2/sizeof(float); a.s3 = nb3/sizeof(float);
    a.tile_spec = tile_spec.get(); a.ids_dst = ids_dst.get(); a.n_used = n_used; a.ne11_src = ne11;
    a.ntiles_m = (ne01 + 16*nwarps - 1)/(16*nwarps);
    a.ntiles_n = ntj;
    a.ntiles   = a.ntiles_m*ntj;

    switch (src0->type) {
        case GGML_TYPE_MS_C1: mul_mat_ms_c_id_switch<1>(ctx, nwarps, bn, acc_f16, a); break;
        case GGML_TYPE_MS_C2: mul_mat_ms_c_id_switch<2>(ctx, nwarps, bn, acc_f16, a); break;
        case GGML_TYPE_MS_C3: mul_mat_ms_c_id_switch<3>(ctx, nwarps, bn, acc_f16, a); break;
        case GGML_TYPE_MS_C4: mul_mat_ms_c_id_switch<4>(ctx, nwarps, bn, acc_f16, a); break;
        case GGML_TYPE_MS_C5: mul_mat_ms_c_id_switch<5>(ctx, nwarps, bn, acc_f16, a); break;
        case GGML_TYPE_MS_C6: mul_mat_ms_c_id_switch<6>(ctx, nwarps, bn, acc_f16, a); break;
        default:
            GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_should_fuse_mul_mat_ms(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * glu, const int cc) {
    if (up->op != GGML_OP_MUL_MAT || gate->op != GGML_OP_MUL_MAT || glu->op != GGML_OP_GLU) {
        return false;
    }
    const ggml_tensor * x_up   = up->src[0];
    const ggml_tensor * x_gate = gate->src[0];
    const int dk = mmms_type_bits(x_up->type) - mmms_type_bits(x_gate->type);
    if (!ggml_cuda_should_use_mmms(x_up->type, cc, up->src[1]->ne[1]) || !ggml_cuda_should_use_mmms(x_gate->type, cc, up->src[1]->ne[1]) ||
            dk < -MMMS_GLU_MAX_DK || dk > MMMS_GLU_MAX_DK ||
            up->src[1] != gate->src[1] || !ggml_are_same_shape(x_up, x_gate) ||
            ggml_get_op_params_i32(up, 1) != GGML_HINT_NONE || ggml_get_op_params_i32(gate, 1) != GGML_HINT_NONE) {
        return false;
    }
    // the 128 column tile only (the batches after the small ones)
    if (up->src[1]->ne[1] <= 64 || up->src[1]->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu) || glu->type != GGML_TYPE_F32) {
        return false;
    }
    const ggml_glu_op op = ggml_get_glu_op(glu);
    return op == GGML_GLU_OP_SWIGLU || op == GGML_GLU_OP_GEGLU;
}

void ggml_cuda_mul_mat_ms_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * up, const ggml_tensor * gate, ggml_tensor * glu) {
    ggml_cuda_mul_mat_ms_impl(ctx, gate->src[0], up->src[0], up->src[1], glu, true, (int) ggml_get_glu_op(glu), ggml_get_op_params_f32(glu, 3));
}
