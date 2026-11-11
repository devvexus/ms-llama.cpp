#include "mmms.cuh"
#include "mmvq.cuh"
#include "cp-async.cuh"

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
    return __dp4a(GGML_MS_C_A*s + GGML_MS_C_B, 0x01010101u, 0x6600u - 510u);
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
    int ne00; int ne01; int64_t s01; int64_t s02; int64_t s03;
    int ne1; int ne11_pad; int ne12; int ne13; int r2; int r3;
    int64_t s1; int64_t s2; int64_t s3;
    int ntiles_m; int ntiles_n; int ntiles; int nblocks;
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

template <int k, int nwarps, int bn, bool acc_f16>
static __global__ void __launch_bounds__(nwarps*WARP_SIZE)
mul_mat_ms_c(const mmms_args a) {
#if defined(TURING_MMA_AVAILABLE)
    using bl = mmms_block<k>;
    constexpr int bm      = 16*nwarps;
    constexpr int nthr    = nwarps*WARP_SIZE;
    constexpr int ntn     = bn/8;                  // n tiles of the mma
    constexpr int npieces = bn*MMMS_BK*2/16;       // 16 byte pieces of a y stage
    constexpr int cpb     = QK_MS/MMMS_BK;         // chunks per block
    constexpr int sbytes  = bn*MMMS_SROW*sizeof(half);
    static_assert(ntn % 2 == 0 && npieces % nthr == 0, "bad tile");

    extern __shared__ half2 tile_y[];
    const unsigned int ty0 = (unsigned int) __cvta_generic_to_shared(tile_y);

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
        const int tn   = tile % a.ntiles_n;
        const int tm   = (tile / a.ntiles_n) % a.ntiles_m;
        const int iz   = tile / (a.ntiles_n*a.ntiles_m);
        const int i12  = iz % a.ne12;
        const int i13  = iz / a.ne12;

        const int col0 = tn*bn;
        const int row0 = tm*bm + 16*warp;

        const char * x0 = a.x + (i13/a.r3)*a.s03 + (i12/a.r2)*a.s02;
        const char * xa = x0 + min(row0 + g + 0, a.ne01 - 1)*a.s01;
        const char * xb = x0 + min(row0 + g + 8, a.ne01 - 1)*a.s01;

        const half2 * yz = a.y + ((int64_t) iz*a.ne11_pad + col0)*(a.ne00/2);

        auto load_y = [&](const int kc) {
            const unsigned int dst_tile = ty0 + (kc % MMMS_NSTAGES)*sbytes;
#pragma unroll
            for (int i = 0; i < npieces/nthr; ++i) {
                const int p    = i*nthr + tid;
                const int col  = p / 8;
                const int part = p % 8;
                mmms_copy_16<(bn >= 64)>(dst_tile + (col*MMMS_SROW + 8*part)*sizeof(half),
                    yz + (int64_t) col*(a.ne00/2) + kc*(MMMS_BK/2) + 4*part);
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

            static_for<0, MMMS_BK/16>([&](auto s_) {
                constexpr int s = decltype(s_)::value;
                uint32_t a0, a1, a2, a3;
                mmms_decode_step<k, s>(wna, dca, a0, a2);
                mmms_decode_step<k, s>(wnb, dcb, a1, a3);

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

        if (kc0 == 0 && kc1 == nchunks) {
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
    GGML_UNUSED(a);
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
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
    return !disabled && GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && ne11 > MMVQ_MAX_BATCH_SIZE;
}

template <int k, int nwarps, int bn, bool acc_f16>
static void mul_mat_ms_c_launch(ggml_backend_cuda_context & ctx, mmms_args a, const bool stream_k) {
    cudaStream_t stream = ctx.stream();
    constexpr size_t nbytes_shared = mmms_nbytes_shared<bn>();
    auto kernel = mul_mat_ms_c<k, nwarps, bn, acc_f16>;
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
    MMMS_CASE(8, 128) MMMS_CASE(8, 64) MMMS_CASE(4, 32) MMMS_CASE(4, 16)
#undef MMMS_CASE
    GGML_ABORT("fatal error");
}

// f16 sums per chunk (added to f32 sums) or f32 sums. The f32 sums are exact up to the f16 inputs (KLD ~1e-6 vs ~4e-6),
// GeForce and TITAN cards run them at half the tensor core rate, data center and workstation cards at full rate.
// GB10 (DGX Spark): f16 sums until its rate is measured. GGML_CUDA_MMMS_ACC=f16|f32 overrides the choice.
static bool mmms_use_f16_sums(const int id, const int cc) {
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
            const bool half_rate = strstr(prop.name, "GeForce") != nullptr || strstr(prop.name, "TITAN") != nullptr ||
                cc == GGML_CUDA_CC_DGX_SPARK;
            choice[id] = half_rate ? 1 : 2;
        }
    }
    return choice[id] == 1;
}

void ggml_cuda_mul_mat_ms(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == sizeof(float));
    GGML_ASSERT(nb0  == sizeof(float));
    GGML_ASSERT(ne00 % QK_MS == 0 && ne00 == ne10);
    GGML_ASSERT(ne12 % ne02 == 0 && ne13 % ne03 == 0);

    cudaStream_t stream = ctx.stream();

    // n tile: the smallest that holds the batch (small batches waste less of the mma on padding columns)
    const int bn     = ne11 <= 16 ? 16 : ne11 <= 32 ? 32 : ne11 <= 64 ? 64 : 128;
    const int nwarps = bn <= 32 ? 4 : 8;

    const int  id       = ggml_cuda_get_device();
    const bool acc_f16  = mmms_use_f16_sums(id, ggml_cuda_info().devices[id].cc);
    const bool stream_k = true;

    const int ne11_pad = GGML_PAD(ne11, bn);

    ggml_cuda_pool_alloc<half2> y(ctx.pool(), ne12*ne13*ne11_pad*ne10/2);
    ggml_cuda_pool_alloc<float> inv_scale(ctx.pool(), ne12*ne13*ne11_pad);
    {
        const bool aligned = (uintptr_t) src1->data % 8 == 0 && nb11 % 8 == 0 && nb12 % 8 == 0 && nb13 % 8 == 0;
        const dim3 grid(ne12*ne13*ne11_pad);
        if (aligned) {
            mmms_convert_y<true><<<grid, MMMS_CONVERT_NT, 0, stream>>>((const float *) src1->data, (uint4 *) y.get(), inv_scale.get(),
                ne10, ne11, ne11_pad, ne12, nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float));
        } else {
            mmms_convert_y<false><<<grid, MMMS_CONVERT_NT, 0, stream>>>((const float *) src1->data, (uint4 *) y.get(), inv_scale.get(),
                ne10, ne11, ne11_pad, ne12, nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    mmms_args a = {};
    a.x = (const char *) src0->data; a.y = y.get(); a.inv_scale = inv_scale.get(); a.dst = (float *) dst->data; a.part = nullptr;
    a.ne00 = ne00; a.ne01 = ne01; a.s01 = nb01; a.s02 = nb02; a.s03 = nb03;
    a.ne1 = ne11; a.ne11_pad = ne11_pad; a.ne12 = ne12; a.ne13 = ne13; a.r2 = ne12/ne02; a.r3 = ne13/ne03;
    a.s1 = nb1/sizeof(float); a.s2 = nb2/sizeof(float); a.s3 = nb3/sizeof(float);

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
