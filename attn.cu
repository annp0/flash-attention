#include <stdint.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

# define WARP_SIZE 32

__device__ inline void __cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n");
}

template<int n>
__device__ inline void __cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" : : "n"(n));
}

template<int n>
__device__ inline void __cp_async_cg(uint32_t dst, const nv_bfloat16* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                 : : "r"(dst), "l"(src), "n"(n));
}

__device__ inline void __ldmatrix_m8n8_x4_b16(uint32_t regs[4], uint32_t src) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
                : "r"(src));
}

__device__ inline void __ldmatrix_m8n8_x4_trans_b16(uint32_t regs[4], uint32_t src) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
                : "r"(src));
}

__device__ inline void __mma_m16n8k16_bf16(uint32_t A[4], uint32_t B[2], float C[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
                : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
                  "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
}

__device__ inline uint32_t swizzle_m8n16_align8_bf16(uint32_t idx){
    return idx ^ ((idx >> 4) & 0x70); 
}

template<int M, int N, int NUM_THREADS>
__device__ inline void cp_g2s_async(uint32_t dst, const nv_bfloat16 *src, int tid){
    constexpr int n = M * N / (NUM_THREADS * 8);
    for (int i = 0; i < n; i++){
        const int idx = (i * NUM_THREADS + tid) * 8;
        const int row = idx / N;
        const int col = idx % N;
        const uint32_t _dst = swizzle_m8n16_align8_bf16(dst + (row * N + col) * sizeof(nv_bfloat16));
        const nv_bfloat16* _src = src + row * N + col;
        __cp_async_cg<16>(_dst, _src);
    }    
}

__global__ __launch_bounds__(128)
void cross_attn(
    int z,
    int len_q,
    int len_kv,
    const nv_bfloat16* ptr_q,
    const nv_bfloat16* ptr_k,
    const nv_bfloat16* ptr_v,
    nv_bfloat16* ptr_o
) {
    constexpr int BLOCK_MAJOR = 64;
    constexpr int BLOCK_MINOR = 32;
    constexpr int HEAD_DIM = 128;
    constexpr int BUF_SIZE = BLOCK_MINOR * HEAD_DIM * sizeof(nv_bfloat16);
    constexpr int NUM_WARPS = 4;
    constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;
    constexpr int WARP_SIZE_Q = BLOCK_MAJOR / NUM_WARPS;
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;

    const float scale = rsqrtf(static_cast<float>(HEAD_DIM));

    const int bid = blockIdx.x;
    const int zid = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    ptr_q += (zid * len_q + bid * BLOCK_MAJOR) * HEAD_DIM;
    ptr_k += zid * len_kv * HEAD_DIM;
    ptr_v += zid * len_kv * HEAD_DIM;
    ptr_o += (zid * len_q + bid * BLOCK_MAJOR) * HEAD_DIM;

    extern __shared__ nv_bfloat16 smem[];
    const uint32_t s_q = __cvta_generic_to_shared(smem);
    const uint32_t s_k = s_q;
    const uint32_t s_v = s_k + BLOCK_MINOR * HEAD_DIM * sizeof(nv_bfloat16);

    uint32_t r_q[WARP_SIZE_Q / MMA_M][HEAD_DIM / MMA_K][4];
    uint32_t r_k[BLOCK_MINOR / MMA_N][HEAD_DIM / MMA_K][2];
    uint32_t r_p[WARP_SIZE_Q / MMA_M][BLOCK_MINOR / MMA_K][4];
    uint32_t r_v[BLOCK_MINOR / MMA_K][HEAD_DIM / MMA_N][2];
    float r_o[WARP_SIZE_Q / MMA_M][HEAD_DIM / MMA_N][4] = {};

    const int q_ldmx_row = warp_id * WARP_SIZE_Q + (lane_id % 16);
    const int q_ldmx_col = (lane_id / 16) * 8;
    const int q_ldmx_src = swizzle_m8n16_align8_bf16(s_q + (q_ldmx_row * HEAD_DIM + q_ldmx_col) * sizeof(nv_bfloat16));
    
    const int k_ldmx_row = lane_id % 8;
    const int k_ldmx_col = (lane_id / 8) * 8;
    const int k_ldmx_src = swizzle_m8n16_align8_bf16(s_k + (k_ldmx_row * HEAD_DIM + k_ldmx_col) * sizeof(nv_bfloat16));
    
    const int v_ldmx_row = lane_id % 16;
    const int v_ldmx_col = (lane_id / 16) * 8;
    const int v_ldmx_src = swizzle_m8n16_align8_bf16(s_v + (v_ldmx_row * HEAD_DIM + v_ldmx_col) * sizeof(nv_bfloat16));

    float max_row[WARP_SIZE_Q / MMA_M][2];
    float expsum_row[WARP_SIZE_Q / MMA_M][2] = {};

    for (int idx_mma_m = 0; idx_mma_m < WARP_SIZE_Q / MMA_M; idx_mma_m++) {
        max_row[idx_mma_m][0] = -FLT_MAX;
        max_row[idx_mma_m][1] = -FLT_MAX;
    }

    cp_g2s_async<BLOCK_MAJOR, HEAD_DIM, NUM_THREADS>(s_q, ptr_q, tid);
    __cp_async_commit_group();
    __cp_async_wait_group<0>();
    __syncthreads();

    for (int idx_mma_m = 0; idx_mma_m < WARP_SIZE_Q / MMA_M; idx_mma_m++) {
        for (int idx_mma_k = 0; idx_mma_k < HEAD_DIM / MMA_K; idx_mma_k++) {
            uint32_t src = q_ldmx_src;
            src += idx_mma_m * MMA_M * HEAD_DIM * sizeof(nv_bfloat16);
            src ^= idx_mma_k * MMA_K * sizeof(nv_bfloat16);
            __ldmatrix_m8n8_x4_b16(r_q[idx_mma_m][idx_mma_k], src);
        }
    }
    __syncthreads();

    cp_g2s_async<BLOCK_MINOR, HEAD_DIM, NUM_THREADS>(s_k, ptr_k, tid);
    __cp_async_commit_group();
    ptr_k += BLOCK_MINOR * HEAD_DIM;

    cp_g2s_async<BLOCK_MINOR, HEAD_DIM, NUM_THREADS>(s_v, ptr_v, tid);
    __cp_async_commit_group();
    ptr_v += BLOCK_MINOR * HEAD_DIM;

    for (int idx_kv = 0; idx_kv < len_kv / BLOCK_MINOR; idx_kv++) {
        float r_s[WARP_SIZE_Q / MMA_M][BLOCK_MINOR / MMA_N][4] = {};

        if (idx_kv < len_kv / BLOCK_MINOR - 1){
            const uint32_t dst = s_k + !(idx_kv & 1) * (2 * BUF_SIZE);
            cp_g2s_async<BLOCK_MINOR, HEAD_DIM, NUM_THREADS>(dst, ptr_k, tid);
            ptr_k += BLOCK_MINOR * HEAD_DIM;
        }
        __cp_async_commit_group();
        __cp_async_wait_group<2>();
        __syncthreads();

        if (idx_kv < len_kv / BLOCK_MINOR - 1){
            const uint32_t dst = s_v + !(idx_kv & 1) * (2 * BUF_SIZE);
            cp_g2s_async<BLOCK_MINOR, HEAD_DIM, NUM_THREADS>(dst, ptr_v, tid);
            ptr_v += BLOCK_MINOR * HEAD_DIM;
        }
        __cp_async_commit_group();

        for (int idx_mma_n = 0; idx_mma_n < BLOCK_MINOR / MMA_N; idx_mma_n++)
            for (int idx_mma_k = 0; idx_mma_k < HEAD_DIM / MMA_K; idx_mma_k += 2) {
                uint32_t src = k_ldmx_src + (idx_kv % 2) * (2 * BLOCK_MINOR * HEAD_DIM * sizeof(nv_bfloat16));
                src += idx_mma_n * MMA_N * HEAD_DIM * sizeof(nv_bfloat16);
                src ^= idx_mma_k * MMA_K * sizeof(nv_bfloat16);
                __ldmatrix_m8n8_x4_b16(r_k[idx_mma_n][idx_mma_k], src);
            }

        for (int idx_mma_m = 0; idx_mma_m < WARP_SIZE_Q / MMA_M; idx_mma_m++){
            for (int idx_mma_n = 0; idx_mma_n < BLOCK_MINOR / MMA_N; idx_mma_n++){
                for (int idx_mma_k = 0; idx_mma_k < HEAD_DIM / MMA_K; idx_mma_k++){
                    __mma_m16n8k16_bf16(r_q[idx_mma_m][idx_mma_k], r_k[idx_mma_n][idx_mma_k], r_s[idx_mma_m][idx_mma_n]);
                }
            }
        }

        for (int idx_mma_m = 0; idx_mma_m < WARP_SIZE_Q / MMA_M; idx_mma_m++) {
            for (int idx_mma_n = 0; idx_mma_n < BLOCK_MINOR / MMA_N; idx_mma_n++)
                for (int reg_id = 0; reg_id < 4; reg_id++)
                    r_s[idx_mma_m][idx_mma_n][reg_id] *= scale;

            float max_row_curr[2] = {-FLT_MAX, -FLT_MAX};
            for (int idx_mma_n = 0; idx_mma_n < BLOCK_MINOR / MMA_N; idx_mma_n++) {
                float *r = r_s[idx_mma_m][idx_mma_n];
                max_row_curr[0] = max(max_row_curr[0], max(r[0], r[1]));
                max_row_curr[1] = max(max_row_curr[1], max(r[2], r[3]));
            }
            max_row_curr[0] = max(max_row_curr[0], __shfl_xor_sync(0xFFFFFFFF, max_row_curr[0], 1));
            max_row_curr[0] = max(max_row_curr[0], __shfl_xor_sync(0xFFFFFFFF, max_row_curr[0], 2));
            max_row_curr[1] = max(max_row_curr[1], __shfl_xor_sync(0xFFFFFFFF, max_row_curr[1], 1));
            max_row_curr[1] = max(max_row_curr[1], __shfl_xor_sync(0xFFFFFFFF, max_row_curr[1], 2));

            max_row_curr[0] = max(max_row_curr[0], max_row[idx_mma_m][0]);
            max_row_curr[1] = max(max_row_curr[1], max_row[idx_mma_m][1]);

            float alpha[2];
            alpha[0] = __expf(max_row[idx_mma_m][0] - max_row_curr[0]);
            alpha[1] = __expf(max_row[idx_mma_m][1] - max_row_curr[1]);

            for (int idx_mma_n = 0; idx_mma_n < HEAD_DIM / MMA_N; idx_mma_n++) {
                r_o[idx_mma_m][idx_mma_n][0] *= alpha[0];
                r_o[idx_mma_m][idx_mma_n][1] *= alpha[0];
                r_o[idx_mma_m][idx_mma_n][2] *= alpha[1];
                r_o[idx_mma_m][idx_mma_n][3] *= alpha[1];
            }

            max_row[idx_mma_m][0] = max_row_curr[0];
            max_row[idx_mma_m][1] = max_row_curr[1];

            float expsum_row_curr[2] = {};
            for (int idx_mma_n = 0; idx_mma_n < BLOCK_MINOR / MMA_N; idx_mma_n++) {
                float *r = r_s[idx_mma_m][idx_mma_n];
                r[0] = __expf(r[0] - max_row[idx_mma_m][0]);
                r[1] = __expf(r[1] - max_row[idx_mma_m][0]);
                r[2] = __expf(r[2] - max_row[idx_mma_m][1]);
                r[3] = __expf(r[3] - max_row[idx_mma_m][1]);
                expsum_row_curr[0] += r[0] + r[1];
                expsum_row_curr[1] += r[2] + r[3];

                nv_bfloat162 *p_pack = reinterpret_cast<nv_bfloat162 *>(r_p[idx_mma_m][idx_mma_n / 2]);
                p_pack[(idx_mma_n % 2) * 2] = __float22bfloat162_rn({r[0], r[1]});
                p_pack[(idx_mma_n % 2) * 2 + 1] = __float22bfloat162_rn({r[2], r[3]});
            }

            expsum_row_curr[0] += __shfl_xor_sync(0xFFFFFFFF, expsum_row_curr[0], 1);
            expsum_row_curr[0] += __shfl_xor_sync(0xFFFFFFFF, expsum_row_curr[0], 2);
            expsum_row_curr[1] += __shfl_xor_sync(0xFFFFFFFF, expsum_row_curr[1], 1);
            expsum_row_curr[1] += __shfl_xor_sync(0xFFFFFFFF, expsum_row_curr[1], 2);

            expsum_row[idx_mma_m][0] = expsum_row[idx_mma_m][0] * alpha[0] + expsum_row_curr[0];
            expsum_row[idx_mma_m][1] = expsum_row[idx_mma_m][1] * alpha[1] + expsum_row_curr[1];
        }

        __cp_async_wait_group<2>();
        __syncthreads();

        for (int idx_mma_k = 0; idx_mma_k < BLOCK_MINOR / MMA_K; idx_mma_k++){
            for (int idx_mma_n = 0; idx_mma_n < HEAD_DIM / MMA_N; idx_mma_n += 2) {
                uint32_t src = v_ldmx_src + (idx_kv % 2) * (2 * BLOCK_MINOR * HEAD_DIM * sizeof(nv_bfloat16));
                src += idx_mma_k * MMA_K * HEAD_DIM * sizeof(nv_bfloat16);
                src ^= idx_mma_n * MMA_N * sizeof(nv_bfloat16);
                __ldmatrix_m8n8_x4_trans_b16(r_v[idx_mma_k][idx_mma_n], src);
            }
        }

        for (int idx_mma_m = 0; idx_mma_m < WARP_SIZE_Q / MMA_M; idx_mma_m++){
            for (int idx_mma_n = 0; idx_mma_n < HEAD_DIM / MMA_N; idx_mma_n++){
                for (int idx_mma_k = 0; idx_mma_k < BLOCK_MINOR / MMA_K; idx_mma_k++){
                    __mma_m16n8k16_bf16(r_p[idx_mma_m][idx_mma_k], r_v[idx_mma_k][idx_mma_n], r_o[idx_mma_m][idx_mma_n]);
                }
            }
        }

    }

    for (int idx_mma_m = 0; idx_mma_m < WARP_SIZE_Q / MMA_M; idx_mma_m++){
        for (int idx_mma_n = 0; idx_mma_n < HEAD_DIM / MMA_N; idx_mma_n++) {
            const int row = warp_id * WARP_SIZE_Q + idx_mma_m * MMA_M + (lane_id / 4);
            const int col = idx_mma_n * MMA_N + (lane_id % 4) * 2;
            float *r = r_o[idx_mma_m][idx_mma_n];
            r[0] /= expsum_row[idx_mma_m][0];
            r[1] /= expsum_row[idx_mma_m][0];
            r[2] /= expsum_row[idx_mma_m][1];
            r[3] /= expsum_row[idx_mma_m][1];
            reinterpret_cast<nv_bfloat162 *>(ptr_o + (row + 0) * HEAD_DIM + col)[0] = __float22bfloat162_rn({r[0], r[1]});
            reinterpret_cast<nv_bfloat162 *>(ptr_o + (row + 8) * HEAD_DIM + col)[0] = __float22bfloat162_rn({r[2], r[3]});
        }
    }
}

#include <torch/types.h>
#include <torch/extension.h>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                    \
if(((T).options().dtype() != (th_type))) {                      \
    std::cout << "Tensor Info:" << (T).options() << std::endl;  \
    throw std::runtime_error("values must be "#th_type);        \
}

at::Tensor torch_cross_attn(
    const at::Tensor& Q,
    const at::Tensor& K,
    const at::Tensor& V
){

    CHECK_TORCH_TENSOR_DTYPE(Q, at::kBFloat16);
    CHECK_TORCH_TENSOR_DTYPE(K, at::kBFloat16);
    CHECK_TORCH_TENSOR_DTYPE(V, at::kBFloat16);

    const int batch_size_q = Q.size(0) * Q.size(1);
    const int batch_size_k = K.size(0) * K.size(1);
    const int batch_size_v = V.size(0) * V.size(1);

    TORCH_CHECK(batch_size_q == batch_size_k, 
                "Error: Batch size of Q and K must match. Got ", batch_size_q, " and ", batch_size_k);
    TORCH_CHECK(batch_size_q == batch_size_v, 
                "Error: Batch size of Q and V must match. Got ", batch_size_q, " and ", batch_size_v);

    const int head_dim_q = Q.size(3);
    const int head_dim_k = K.size(3);
    const int head_dim_v = V.size(3);

    TORCH_CHECK(head_dim_q == 128, 
                "Error: Head dimension of Q must be 128. Got ", head_dim_q);
    TORCH_CHECK(head_dim_k == 128, 
                "Error: Head dimension of K must be 128. Got ", head_dim_k);
    TORCH_CHECK(head_dim_v == 128, 
                "Error: Head dimension of V must be 128. Got ", head_dim_v);
    TORCH_CHECK(head_dim_q == head_dim_k && head_dim_q == head_dim_v, 
                "Error: Head dimensions of Q, K, and V must match.");

    const int z = batch_size_q;
    const int len_q = Q.size(2);
    const int len_kv = K.size(2);

    TORCH_CHECK(len_q % 64 == 0, 
                "Error: len_q must be divisible by 64. Got len_q = ", len_q);
    TORCH_CHECK(len_kv % 32 == 0, 
                "Error: len_kv must be divisible by 32. Got len_kv = ", len_kv);

    at::Tensor O = at::empty_like(Q);

    auto ptr_q = reinterpret_cast<const nv_bfloat16 *>(Q.data_ptr());
    auto ptr_k = reinterpret_cast<const nv_bfloat16 *>(K.data_ptr());
    auto ptr_v = reinterpret_cast<const nv_bfloat16 *>(V.data_ptr());
    auto ptr_o = reinterpret_cast<nv_bfloat16 *>(O.data_ptr());

    cross_attn<<<dim3(len_q / 64, z), 128, 32768>>>(z, len_q, len_kv, ptr_q, ptr_k, ptr_v, ptr_o);

    return O;
}

