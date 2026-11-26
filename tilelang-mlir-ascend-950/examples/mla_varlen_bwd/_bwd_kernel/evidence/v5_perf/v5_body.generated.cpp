#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/nd2nz_copy.h>
__simt_vf__ __launch_bounds__(128) inline void _bwd_main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  #pragma unroll
  for (int32_t i = 0; i < 12; ++i) {
    float broadcast_var = float(0x1.279a74590331cp-4f/*7.216878e-02*/);
    float2 _;
      float2 v_ = (*(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) + (((i * 256) + (((int32_t)threadIdx.x) * 2)) + 19968)) * make_float2(broadcast_var, broadcast_var)) - *(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) + (((((int32_t)threadIdx.x) & 31) * 2) + 38528));
      _.x = expf(v_.x);
      _.y = expf(v_.y);
    float2 p = _;
    uint1 __1;
      (reinterpret_cast<bfloat16x2_t*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&p))[0]);
    *(__ubuf__ uint1*)(((__ubuf__ bfloat16_t*)buf_dyn_shmem) + (((i * 256) + (((int32_t)threadIdx.x) * 2)) + 70912)) = __1;
    uint1 __2;
      float2 v__1 = p * (*(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) + (((i * 256) + (((int32_t)threadIdx.x) * 2)) + 23040)) - *(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) + (((((int32_t)threadIdx.x) & 31) * 2) + 38592)));
      (reinterpret_cast<bfloat16x2_t*>(&__2))[0] = __float22bfloat162_rn(((float2*)(&v__1))[0]);
    *(__ubuf__ uint1*)(((__ubuf__ bfloat16_t*)buf_dyn_shmem) + (((i * 256) + (((int32_t)threadIdx.x) * 2)) + 73984)) = __2;
  }
}

__simt_vf__ __launch_bounds__(128) inline void _bwd_main_kernel_simt_vf_1(int64_t sid_1, int32_t n0_1, int32_t len_kv_1, int32_t i_1, int32_t len_q_1, __ubuf__ uint8_t* buf_dyn_shmem) {
  #pragma unroll
  for (int32_t i = 0; i < 24; ++i) {
    bool vis = ((((((((int32_t)sid_1) * 48) + (i * 2)) + (((int32_t)threadIdx.x) >> 6)) + n0_1) < len_kv_1) & ((((i_1 * 128) + (max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6)) * 64)) + (((int32_t)threadIdx.x) & 63)) < len_q_1));
    bool vis_1 = (vis & (((((((int32_t)sid_1) * 48) + (i * 2)) + (((int32_t)threadIdx.x) >> 6)) + n0_1) <= (((((i_1 * 128) + (max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6)) * 64)) + len_kv_1) + (((int32_t)threadIdx.x) & 63)) - len_q_1)));
    float p = expf(((((__ubuf__ float*)buf_dyn_shmem)[(((i * 128) + ((int32_t)threadIdx.x)) + 19968)] * float(0x1.279a74590331cp-4f/*7.216878e-02*/)) - ((__ubuf__ float*)buf_dyn_shmem)[((((int32_t)threadIdx.x) & 63) + 38528)]));
    float condval;
    if (vis_1) {
      condval = p;
    } else {
      condval = float(0x0p+0f/*0.000000e+00*/);
    }
    ((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((i * 128) + ((int32_t)threadIdx.x)) + 70912)] = ((bfloat16_t)condval);
    float condval_1;
    if (vis_1) {
      condval_1 = p;
    } else {
      condval_1 = float(0x0p+0f/*0.000000e+00*/);
    }
    ((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((i * 128) + ((int32_t)threadIdx.x)) + 73984)] = ((bfloat16_t)(condval_1 * (((__ubuf__ float*)buf_dyn_shmem)[(((i * 128) + ((int32_t)threadIdx.x)) + 23040)] - ((__ubuf__ float*)buf_dyn_shmem)[((((int32_t)threadIdx.x) & 63) + 38592)])));
  }
}

__simt_vf__ __launch_bounds__(128) inline void _bwd_main_kernel_simt_vf_2(__ubuf__ uint8_t* buf_dyn_shmem) {
  #pragma unroll
  for (int32_t i = 0; i < 12; ++i) {
    float broadcast_var = float(0x1.279a74590331cp-4f/*7.216878e-02*/);
    float2 _;
      float2 v_ = (*(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) + (((i * 256) + (((int32_t)threadIdx.x) * 2)) + 19968)) * make_float2(broadcast_var, broadcast_var)) - *(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) + (((((int32_t)threadIdx.x) & 31) * 2) + 38656));
      _.x = expf(v_.x);
      _.y = expf(v_.y);
    float2 p = _;
    uint1 __1;
      (reinterpret_cast<bfloat16x2_t*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&p))[0]);
    *(__ubuf__ uint1*)(((__ubuf__ bfloat16_t*)buf_dyn_shmem) + (((i * 256) + (((int32_t)threadIdx.x) * 2)) + 70912)) = __1;
    uint1 __2;
      float2 v__1 = p * (*(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) + (((i * 256) + (((int32_t)threadIdx.x) * 2)) + 23040)) - *(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) + (((((int32_t)threadIdx.x) & 31) * 2) + 38720)));
      (reinterpret_cast<bfloat16x2_t*>(&__2))[0] = __float22bfloat162_rn(((float2*)(&v__1))[0]);
    *(__ubuf__ uint1*)(((__ubuf__ bfloat16_t*)buf_dyn_shmem) + (((i * 256) + (((int32_t)threadIdx.x) * 2)) + 73984)) = __2;
  }
}

__simt_vf__ __launch_bounds__(128) inline void _bwd_main_kernel_simt_vf_3(int64_t sid_1, int32_t n0_1, int32_t len_kv_1, int32_t i_1, int32_t len_q_1, __ubuf__ uint8_t* buf_dyn_shmem) {
  #pragma unroll
  for (int32_t i = 0; i < 24; ++i) {
    bool vis = ((((((((int32_t)sid_1) * 48) + (i * 2)) + (((int32_t)threadIdx.x) >> 6)) + n0_1) < len_kv_1) & (((((i_1 * 128) + (max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6)) * 64)) + (((int32_t)threadIdx.x) & 63)) + 64) < len_q_1));
    bool vis_1 = (vis & (((((((int32_t)sid_1) * 48) + (i * 2)) + (((int32_t)threadIdx.x) >> 6)) + n0_1) <= ((((((i_1 * 128) + (max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6)) * 64)) + len_kv_1) + (((int32_t)threadIdx.x) & 63)) + 64) - len_q_1)));
    float p = expf(((((__ubuf__ float*)buf_dyn_shmem)[(((i * 128) + ((int32_t)threadIdx.x)) + 19968)] * float(0x1.279a74590331cp-4f/*7.216878e-02*/)) - ((__ubuf__ float*)buf_dyn_shmem)[((((int32_t)threadIdx.x) & 63) + 38656)]));
    float condval;
    if (vis_1) {
      condval = p;
    } else {
      condval = float(0x0p+0f/*0.000000e+00*/);
    }
    ((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((i * 128) + ((int32_t)threadIdx.x)) + 70912)] = ((bfloat16_t)condval);
    float condval_1;
    if (vis_1) {
      condval_1 = p;
    } else {
      condval_1 = float(0x0p+0f/*0.000000e+00*/);
    }
    ((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((i * 128) + ((int32_t)threadIdx.x)) + 73984)] = ((bfloat16_t)(condval_1 * (((__ubuf__ float*)buf_dyn_shmem)[(((i * 128) + ((int32_t)threadIdx.x)) + 23040)] - ((__ubuf__ float*)buf_dyn_shmem)[((((int32_t)threadIdx.x) & 63) + 38720)])));
  }
}

__simt_vf__ __launch_bounds__(128) inline void _bwd_main_kernel_simt_vf_4(__ubuf__ uint8_t* buf_dyn_shmem) {
  #pragma unroll
  for (int32_t i = 0; i < 36; ++i) {
    bfloat16_t out_k_local_cast[2];
    float tail_u_local_cast_1[2];
    *(float2*)(tail_u_local_cast_1 + 0) = *(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) + ((i * 256) + (((int32_t)threadIdx.x) * 2)));
    float broadcast_var = float(0x1.279a74590331cp-4f/*7.216878e-02*/);
    uint1 _;
      float2 v_ = *(float2*)(tail_u_local_cast_1 + 0) * make_float2(broadcast_var, broadcast_var);
      (reinterpret_cast<bfloat16x2_t*>(&_))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
    *(uint1*)(out_k_local_cast + 0) = _;
    *(__ubuf__ uint1*)(((__ubuf__ bfloat16_t*)buf_dyn_shmem) + (((i * 256) + (((int32_t)threadIdx.x) * 2)) + 30720)) = *(uint1*)(out_k_local_cast + 0);
  }
  #pragma unroll
  for (int32_t i_1 = 0; i_1 < 24; ++i_1) {
    bfloat16_t out_v_local_cast_2[2];
    float tailv_u_local_cast_3[2];
    *(float2*)(tailv_u_local_cast_3 + 0) = *(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) + (((i_1 * 256) + (((int32_t)threadIdx.x) * 2)) + 9216));
    uint1 __1;
      float2 v__1 = *(float2*)(tailv_u_local_cast_3 + 0);
      (reinterpret_cast<bfloat16x2_t*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v__1))[0]);
    *(uint1*)(out_v_local_cast_2 + 0) = __1;
    *(__ubuf__ uint1*)(((__ubuf__ bfloat16_t*)buf_dyn_shmem) + (((i_1 * 256) + (((int32_t)threadIdx.x) * 2)) + 52224)) = *(uint1*)(out_v_local_cast_2 + 0);
  }
}

__simt_vf__ __launch_bounds__(128) inline void _bwd_main_kernel_simt_vf_5(int32_t h_1, int64_t sid_1, int32_t kv_start_1, int32_t n0_1, int32_t len_kv_1, int32_t total_kv, __gm__ bfloat16_t* dk, __ubuf__ uint8_t* buf_dyn_shmem, __gm__ bfloat16_t* dv) {
  if (0 <= h_1) {
    #pragma unroll
    for (int32_t i = 0; i < 72; ++i) {
      if (((((((int32_t)sid_1) * 48) + (((i * 2) + (((int32_t)threadIdx.x) >> 6)) / 3)) + kv_start_1) + n0_1) < (kv_start_1 + len_kv_1)) {
        if (0 <= ((((((int32_t)sid_1) * 48) + (((i * 2) + (((int32_t)threadIdx.x) >> 6)) / 3)) + kv_start_1) + n0_1)) {
          if (((((((int32_t)sid_1) * 48) + (((i * 2) + (((int32_t)threadIdx.x) >> 6)) / 3)) + kv_start_1) + n0_1) < total_kv) {
            dk[((((((sid_1 * (int64_t)18432) + ((((((int64_t)i) * (int64_t)2) + (((int64_t)((int32_t)threadIdx.x)) >> (int64_t)6)) / (int64_t)3) * (int64_t)384)) + (((int64_t)kv_start_1) * (int64_t)384)) + (((int64_t)n0_1) * (int64_t)384)) + (((int64_t)h_1) * (int64_t)192)) + (((((int64_t)i) * (int64_t)128) + ((int64_t)((int32_t)threadIdx.x))) % (int64_t)192))] = ((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((i * 128) + ((int32_t)threadIdx.x)) + 30720)];
          }
        }
      }
    }
    #pragma unroll
    for (int32_t i_1 = 0; i_1 < 48; ++i_1) {
      if (((((((int32_t)sid_1) * 48) + kv_start_1) + n0_1) + i_1) < (kv_start_1 + len_kv_1)) {
        if (0 <= ((((((int32_t)sid_1) * 48) + kv_start_1) + n0_1) + i_1)) {
          if (((((((int32_t)sid_1) * 48) + kv_start_1) + n0_1) + i_1) < total_kv) {
            dv[((((((sid_1 * (int64_t)12288) + (((int64_t)kv_start_1) * (int64_t)256)) + (((int64_t)n0_1) * (int64_t)256)) + (((int64_t)i_1) * (int64_t)256)) + (((int64_t)h_1) * (int64_t)128)) + ((int64_t)((int32_t)threadIdx.x)))] = ((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((((int64_t)i_1) * (int64_t)128) + ((int64_t)((int32_t)threadIdx.x))) + (int64_t)52224)];
          }
        }
      }
    }
  }
}

extern "C" __global__ __mix__(1, 2) void _bwd_main_kernel(__gm__ int32_t* cu_kv, __gm__ int32_t* cu_q, __gm__ float* delta, __gm__ bfloat16_t* dk, __gm__ bfloat16_t* dout, __gm__ float* dq_acc, __gm__ bfloat16_t* dv, __gm__ int32_t* h_off, __gm__ bfloat16_t* k, __gm__ float* lse, __gm__ bfloat16_t* q, __gm__ bfloat16_t* v, int32_t total_kv, int32_t total_q) {
  asc_init();
  __cbuf__ uint8_t *buf_dyn_l1 = (__cbuf__ uint8_t *)0;
  __cc__ uint8_t *buf_dyn_l0c = (__cc__ uint8_t *)0;
  __cb__ uint8_t *buf_dyn_l0b = (__cb__ uint8_t *)0;
  __ca__ uint8_t *buf_dyn_l0a = (__ca__ uint8_t *)0;
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  int32_t cnt[1];
  int32_t req[1];
  int32_t n0_l[1];
  int32_t live[1];
  if ASC_IS_AIC {
    asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>(0));
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(0));
    asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>(1));
    asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>(2));
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(1));
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(2));
    int32_t h = (h_off[0] + (((int32_t)block_idx) & 1));
    if (h < 2) {
      cnt[0] = 0;
      req[0] = 0;
      n0_l[0] = 0;
      for (int32_t g = 0; g < 2; ++g) {
        int32_t nt = ((((cu_kv[(g + 1)] + 95) - cu_kv[g]) / 96) + ((((cu_kv[(g + 1)] + 95) - cu_kv[g]) % 96) >> 31));
        bool __cond_0 = ((cnt[0] <= (((int32_t)block_idx) >> 1)) & ((((int32_t)block_idx) >> 1) < (cnt[0] + nt)));
        if (__cond_0) {
          req[0] = g;
          n0_l[0] = (((((int32_t)block_idx) >> 1) * 96) - (cnt[0] * 96));
        }
        cnt[0] = (cnt[0] + nt);
      }
      int32_t condval;
      if (((((int32_t)block_idx) >> 1) < cnt[0])) {
        condval = 1;
      } else {
        condval = 0;
      }
      live[0] = condval;
    }
    int8_t __cond_5 = ((int8_t)(bool)0);
    if (h < 2) {
      __cond_5 = ((int8_t)(live[0] == 1));
    }
    int32_t n0 = 0;
    if (h < 2) {
      if ((bool)__cond_5) {
        n0 = n0_l[0];
      }
    }
    int32_t q_start = 0;
    if (h < 2) {
      if ((bool)__cond_5) {
        int32_t condval_1;
        if (((0 <= req[0]) && (req[0] < 3))) {
          condval_1 = cu_q[req[(int64_t)0]];
        } else {
          condval_1 = 0;
        }
        q_start = condval_1;
      }
    }
    int32_t kv_start = 0;
    if (h < 2) {
      if ((bool)__cond_5) {
        int32_t condval_2;
        if (((0 <= req[0]) && (req[0] < 3))) {
          condval_2 = cu_kv[req[(int64_t)0]];
        } else {
          condval_2 = 0;
        }
        kv_start = condval_2;
      }
    }
    int32_t len_q = 0;
    if (h < 2) {
      if ((bool)__cond_5) {
        int32_t condval_3;
        if (((-1 <= req[0]) && (req[0] < 2))) {
          condval_3 = cu_q[(((int64_t)req[(int64_t)0]) + (int64_t)1)];
        } else {
          condval_3 = 0;
        }
        len_q = (condval_3 - q_start);
      }
    }
    int32_t len_kv = 0;
    if (h < 2) {
      if ((bool)__cond_5) {
        int32_t condval_4;
        if (((-1 <= req[0]) && (req[0] < 2))) {
          condval_4 = cu_kv[(((int64_t)req[(int64_t)0]) + (int64_t)1)];
        } else {
          condval_4 = 0;
        }
        len_kv = (condval_4 - kv_start);
      }
      if ((bool)__cond_5) {
        if (0 < ((total_kv - kv_start) - n0)) {
          asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(96), 0);
          asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1), (&(k[(((((int64_t)kv_start) * (int64_t)384) + (((int64_t)n0) * (int64_t)384)) + (((int64_t)h) * (int64_t)192))])), 768, static_cast<asc_load_l2_cache_mode>(0), min(96, max(((total_kv - kv_start) - n0), 0)), 192, 0, 0);
        }
      }
      if ((bool)__cond_5) {
        asc_fill_l1((__cbuf__ uint16_t*)((__cbuf__ uint8_t*)((__cbuf__ bfloat16_t*)buf_dyn_l1) + (min(96, max(((total_kv - kv_start) - n0), 0)) * 32)), (uint32_t)((uint32_t)0), { .repeat = static_cast<uint64_t>(12), .blk_num = static_cast<uint64_t>(max(0, (96 - max(((total_kv - kv_start) - n0), 0)))), .dst_gap = static_cast<uint64_t>((96 - max(0, (96 - max(((total_kv - kv_start) - n0), 0)))))});
      }
      if ((bool)__cond_5) {
        if (0 < ((total_kv - kv_start) - n0)) {
          asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(96), 0);
          asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + 36864), (&(v[(((((int64_t)kv_start) * (int64_t)256) + (((int64_t)n0) * (int64_t)256)) + (((int64_t)h) * (int64_t)128))])), 512, static_cast<asc_load_l2_cache_mode>(0), min(96, max(((total_kv - kv_start) - n0), 0)), 128, 0, 0);
        }
      }
      if ((bool)__cond_5) {
        asc_fill_l1((__cbuf__ uint16_t*)((__cbuf__ uint8_t*)((__cbuf__ bfloat16_t*)buf_dyn_l1 + 36864) + (min(96, max(((total_kv - kv_start) - n0), 0)) * 32)), (uint32_t)((uint32_t)0), { .repeat = static_cast<uint64_t>(8), .blk_num = static_cast<uint64_t>(max(0, (96 - max(((total_kv - kv_start) - n0), 0)))), .dst_gap = static_cast<uint64_t>((96 - max(0, (96 - max(((total_kv - kv_start) - n0), 0)))))});
      }
      if ((bool)__cond_5) {
        if (0 < ((total_kv - kv_start) - n0)) {
          asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(192), 0);
          asc_copy_gm2l1_dn2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + 18432), (&(k[(((((int64_t)kv_start) * (int64_t)384) + (((int64_t)n0) * (int64_t)384)) + (((int64_t)h) * (int64_t)192))])), 768, static_cast<asc_load_l2_cache_mode>(0), 192, min(96, max(((total_kv - kv_start) - n0), 0)), 0, 0);
        }
      }
      if ((bool)__cond_5) {
        asc_fill_l1((__cbuf__ uint16_t*)((__cbuf__ uint8_t*)((__cbuf__ bfloat16_t*)buf_dyn_l1 + 18432) + (((min(96, max(((total_kv - kv_start) - n0), 0)) + 15) >> 4) * 6144)), (uint32_t)((uint32_t)0), { .repeat = static_cast<uint64_t>((6 - ((min(96, max(((total_kv - kv_start) - n0), 0)) + 15) >> 4))), .blk_num = static_cast<uint64_t>(192), .dst_gap = static_cast<uint64_t>(0)});
      }
    }
    asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(6));
    asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(6));
    if (h < 2) {
      if ((bool)__cond_5) {
        for (int32_t i = 0; i < ((max((((len_q + 63) >> 6) - max(0, (((n0 + len_q) - len_kv) >> 6))), 1) + 1) >> 1); ++i) {
          if (0 < (((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128))) {
            asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(64), 0);
            asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + 49152), (&(q[((((((int64_t)i) * (int64_t)49152) + (max((int64_t)0, (((((int64_t)n0) + ((int64_t)len_q)) - ((int64_t)len_kv)) >> (int64_t)6)) * (int64_t)24576)) + (((int64_t)q_start) * (int64_t)384)) + (((int64_t)h) * (int64_t)192))])), 768, static_cast<asc_load_l2_cache_mode>(0), min(64, max((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)), 0)), 192, 0, 0);
          }
          asc_fill_l1((__cbuf__ uint16_t*)((__cbuf__ uint8_t*)((__cbuf__ bfloat16_t*)buf_dyn_l1 + 49152) + (min(64, max((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)), 0)) * 32)), (uint32_t)((uint32_t)0), { .repeat = static_cast<uint64_t>(12), .blk_num = static_cast<uint64_t>(max(0, (64 - max((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)), 0)))), .dst_gap = static_cast<uint64_t>((64 - max(0, (64 - max((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)), 0)))))});
          asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(4));
          if (0 < (((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128))) {
            asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(64), 0);
            asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + 98304), (&(dout[((((((int64_t)i) * (int64_t)32768) + (max((int64_t)0, (((((int64_t)n0) + ((int64_t)len_q)) - ((int64_t)len_kv)) >> (int64_t)6)) * (int64_t)16384)) + (((int64_t)q_start) * (int64_t)256)) + (((int64_t)h) * (int64_t)128))])), 512, static_cast<asc_load_l2_cache_mode>(0), min(64, max((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)), 0)), 128, 0, 0);
          }
          asc_fill_l1((__cbuf__ uint16_t*)((__cbuf__ uint8_t*)((__cbuf__ bfloat16_t*)buf_dyn_l1 + 98304) + (min(64, max((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)), 0)) * 32)), (uint32_t)((uint32_t)0), { .repeat = static_cast<uint64_t>(8), .blk_num = static_cast<uint64_t>(max(0, (64 - max((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)), 0)))), .dst_gap = static_cast<uint64_t>((64 - max(0, (64 - max((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)), 0)))))});
          asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(3));
          asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(0));
          if (0 < (((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128))) {
            asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(192), 0);
            asc_copy_gm2l1_dn2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + 61440), (&(q[((((((int64_t)i) * (int64_t)49152) + (max((int64_t)0, (((((int64_t)n0) + ((int64_t)len_q)) - ((int64_t)len_kv)) >> (int64_t)6)) * (int64_t)24576)) + (((int64_t)q_start) * (int64_t)384)) + (((int64_t)h) * (int64_t)192))])), 768, static_cast<asc_load_l2_cache_mode>(0), 192, min(64, max((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)), 0)), 0, 0);
          }
          asc_fill_l1((__cbuf__ uint16_t*)((__cbuf__ uint8_t*)((__cbuf__ bfloat16_t*)buf_dyn_l1 + 61440) + (((min(64, max((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)), 0)) + 15) >> 4) * 6144)), (uint32_t)((uint32_t)0), { .repeat = static_cast<uint64_t>((4 - ((min(64, max((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)), 0)) + 15) >> 4))), .blk_num = static_cast<uint64_t>(192), .dst_gap = static_cast<uint64_t>(0)});
          if (0 < (((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128))) {
            asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(128), 0);
            asc_copy_gm2l1_dn2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + 106496), (&(dout[((((((int64_t)i) * (int64_t)32768) + (max((int64_t)0, (((((int64_t)n0) + ((int64_t)len_q)) - ((int64_t)len_kv)) >> (int64_t)6)) * (int64_t)16384)) + (((int64_t)q_start) * (int64_t)256)) + (((int64_t)h) * (int64_t)128))])), 512, static_cast<asc_load_l2_cache_mode>(0), 128, min(64, max((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)), 0)), 0, 0);
          }
          asc_fill_l1((__cbuf__ uint16_t*)((__cbuf__ uint8_t*)((__cbuf__ bfloat16_t*)buf_dyn_l1 + 106496) + (((min(64, max((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)), 0)) + 15) >> 4) * 4096)), (uint32_t)((uint32_t)0), { .repeat = static_cast<uint64_t>((4 - ((min(64, max((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)), 0)) + 15) >> 4))), .blk_num = static_cast<uint64_t>(128), .dst_gap = static_cast<uint64_t>(0)});
          asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(2));
          asc_copy_l12l0a(((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cbuf__ bfloat16_t*)buf_dyn_l1), 0, 0, 6, 12, 6, 6);
          asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(4));
          asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b + 18432), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 49152), 0, 0, 4, 12, 4, 4);
          asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(1));
          asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>(2));
          asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(1));
          asc_mmad(((__cc__ float*)buf_dyn_l0c + 43008), ((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cb__ bfloat16_t*)buf_dyn_l0b + 18432), 96, 192, 64, 0, 1, 0, (bool)1);
          asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(1));
          asc_sync_notify(PIPE_M, PIPE_FIX, static_cast<event_t>(5));
          asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(2));
          asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(2));
          asc_copy_l12l0a(((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 36864), 0, 0, 6, 8, 6, 6);
          asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(3));
          asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(1));
          asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b + 18432), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 98304), 0, 0, 4, 8, 4, 4);
          asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(1));
          asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>(1));
          asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(1));
          asc_mmad(((__cc__ float*)buf_dyn_l0c + 49152), ((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cb__ bfloat16_t*)buf_dyn_l0b + 18432), 96, 128, 64, 0, 1, 0, (bool)1);
          asc_sync_notify(PIPE_M, PIPE_FIX, static_cast<event_t>(4));
          asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
          asc_sync_intra_wait(PIPE_FIX, 18);
          asc_sync_intra_wait(PIPE_FIX, 2);
          asc_sync_wait(PIPE_M, PIPE_FIX, static_cast<event_t>(5));
          asc_set_l0c_copy_nz_para(1, 0, 0);
          asc_copy_l0c2ub((&(((__ubuf__ float*)buf_dyn_shmem)[19968])), ((__cc__ float*)buf_dyn_l0c + 43008), 64, 96, 64, 96, 0, static_cast<asc_dual_dst_mode>(1), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
          asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>(5));
          asc_sync_wait(PIPE_M, PIPE_FIX, static_cast<event_t>(4));
          asc_set_l0c_copy_nz_para(1, 0, 0);
          asc_copy_l0c2ub((&(((__ubuf__ float*)buf_dyn_shmem)[23040])), ((__cc__ float*)buf_dyn_l0c + 49152), 64, 96, 64, 96, 0, static_cast<asc_dual_dst_mode>(1), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
          asc_sync_intra_arrive(PIPE_FIX, 9);
          asc_sync_intra_arrive(PIPE_FIX, 25);
          asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>(4));
          asc_sync_intra_wait(PIPE_MTE1, 17);
          asc_sync_intra_wait(PIPE_MTE1, 1);
          asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
          if (0 < i) {
            asc_copy_l12l0a(((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 143360), 0, 0, 6, 4, 6, 6);
          }
          asc_sync_intra_arrive(PIPE_MTE1, 5);
          asc_sync_intra_arrive(PIPE_MTE1, 21);
          if (0 < i) {
            asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 122880), 0, 0, 8, 4, 8, 8);
          }
          asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
          asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
          if (0 < i) {
            asc_mmad(((__cc__ float*)buf_dyn_l0c + 30720), ((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cb__ bfloat16_t*)buf_dyn_l0b), 96, 64, 128, 0, 1, 0, (bool)0);
          }
          asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
          asc_sync_intra_wait(PIPE_MTE1, 16);
          asc_sync_intra_wait(PIPE_MTE1, 0);
          asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
          if (0 < i) {
            asc_copy_l12l0a(((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 149504), 0, 0, 6, 4, 6, 6);
            asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 86016), 0, 0, 12, 4, 12, 12);
          }
          asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
          asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(3));
          asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
          if (0 < i) {
            asc_mmad(((__cc__ float*)buf_dyn_l0c), ((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cb__ bfloat16_t*)buf_dyn_l0b), 96, 64, 192, 0, 1, 0, (bool)0);
          }
          asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
          if (0 < i) {
            asc_copy_l12l0a_transpose(((__ca__ bfloat16_t*)buf_dyn_l0a + 18432), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 149504), 0, 0, 6, 4, 6, 4);
          }
          asc_sync_intra_arrive(PIPE_MTE1, 4);
          asc_sync_intra_arrive(PIPE_MTE1, 20);
          asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
          if (0 < i) {
            asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 18432), 0, 0, 12, 6, 12, 12);
          }
          asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
          asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>(0));
          asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
          if (0 < i) {
            asc_mmad(((__cc__ float*)buf_dyn_l0c + 18432), ((__ca__ bfloat16_t*)buf_dyn_l0a + 18432), ((__cb__ bfloat16_t*)buf_dyn_l0b), 64, 96, 192, 0, 1, 0, (bool)1);
          }
          asc_sync_notify(PIPE_M, PIPE_FIX, static_cast<event_t>(3));
          if (0 < i) {
            asc_set_atomic_add_float();
          }
          asc_sync_wait(PIPE_M, PIPE_FIX, static_cast<event_t>(3));
          if (0 < i) {
            if (-64 < (((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128))) {
              asc_set_l0c_copy_nz_para(1, 0, 0);
              asc_copy_l0c2gm((&(dq_acc[(((((((int64_t)i) * (int64_t)49152) + (max((int64_t)0, (((((int64_t)n0) + ((int64_t)len_q)) - ((int64_t)len_kv)) >> (int64_t)6)) * (int64_t)24576)) + (((int64_t)q_start) * (int64_t)384)) + (((int64_t)h) * (int64_t)192)) - (int64_t)24576)])), ((__cc__ float*)buf_dyn_l0c + 18432), 192, min(64, max(((((total_q + 64) - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)), 0)), 384, 64, static_cast<asc_store_l2_cache_mode>(0), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
            }
          }
          asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>(3));
          if (0 < i) {
            asc_set_atomic_none();
          }
          bool __cond_2 = ((((i * 2) + max(0, (((n0 + len_q) - len_kv) >> 6))) + 1) < ((len_q + 63) >> 6));
          if (__cond_2) {
            asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(2));
            if (64 < (((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128))) {
              asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(64), 0);
              asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + 73728), (&(q[(((((((int64_t)i) * (int64_t)49152) + (max((int64_t)0, (((((int64_t)n0) + ((int64_t)len_q)) - ((int64_t)len_kv)) >> (int64_t)6)) * (int64_t)24576)) + (((int64_t)q_start) * (int64_t)384)) + (((int64_t)h) * (int64_t)192)) + (int64_t)24576)])), 768, static_cast<asc_load_l2_cache_mode>(0), min(64, max(((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)) - 64), 0)), 192, 0, 0);
            }
            asc_fill_l1((__cbuf__ uint16_t*)((__cbuf__ uint8_t*)((__cbuf__ bfloat16_t*)buf_dyn_l1 + 73728) + (min(64, max(((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)) - 64), 0)) * 32)), (uint32_t)((uint32_t)0), { .repeat = static_cast<uint64_t>(12), .blk_num = static_cast<uint64_t>(max(0, (64 - max(((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)) - 64), 0)))), .dst_gap = static_cast<uint64_t>((64 - max(0, (64 - max(((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)) - 64), 0)))))});
            asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(1));
            asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(1));
            if (64 < (((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128))) {
              asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(64), 0);
              asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + 114688), (&(dout[(((((((int64_t)i) * (int64_t)32768) + (max((int64_t)0, (((((int64_t)n0) + ((int64_t)len_q)) - ((int64_t)len_kv)) >> (int64_t)6)) * (int64_t)16384)) + (((int64_t)q_start) * (int64_t)256)) + (((int64_t)h) * (int64_t)128)) + (int64_t)16384)])), 512, static_cast<asc_load_l2_cache_mode>(0), min(64, max(((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)) - 64), 0)), 128, 0, 0);
            }
            asc_fill_l1((__cbuf__ uint16_t*)((__cbuf__ uint8_t*)((__cbuf__ bfloat16_t*)buf_dyn_l1 + 114688) + (min(64, max(((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)) - 64), 0)) * 32)), (uint32_t)((uint32_t)0), { .repeat = static_cast<uint64_t>(8), .blk_num = static_cast<uint64_t>(max(0, (64 - max(((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)) - 64), 0)))), .dst_gap = static_cast<uint64_t>((64 - max(0, (64 - max(((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)) - 64), 0)))))});
            asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));
          }
          asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(3));
          if (__cond_2) {
            if (64 < (((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128))) {
              asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(192), 0);
              asc_copy_gm2l1_dn2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + 86016), (&(q[(((((((int64_t)i) * (int64_t)49152) + (max((int64_t)0, (((((int64_t)n0) + ((int64_t)len_q)) - ((int64_t)len_kv)) >> (int64_t)6)) * (int64_t)24576)) + (((int64_t)q_start) * (int64_t)384)) + (((int64_t)h) * (int64_t)192)) + (int64_t)24576)])), 768, static_cast<asc_load_l2_cache_mode>(0), 192, min(64, max(((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)) - 64), 0)), 0, 0);
            }
            asc_fill_l1((__cbuf__ uint16_t*)((__cbuf__ uint8_t*)((__cbuf__ bfloat16_t*)buf_dyn_l1 + 86016) + (((min(64, max(((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)) - 64), 0)) + 15) >> 4) * 6144)), (uint32_t)((uint32_t)0), { .repeat = static_cast<uint64_t>((4 - ((min(64, max(((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)) - 64), 0)) + 15) >> 4))), .blk_num = static_cast<uint64_t>(192), .dst_gap = static_cast<uint64_t>(0)});
            if (64 < (((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128))) {
              asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(128), 0);
              asc_copy_gm2l1_dn2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + 122880), (&(dout[(((((((int64_t)i) * (int64_t)32768) + (max((int64_t)0, (((((int64_t)n0) + ((int64_t)len_q)) - ((int64_t)len_kv)) >> (int64_t)6)) * (int64_t)16384)) + (((int64_t)q_start) * (int64_t)256)) + (((int64_t)h) * (int64_t)128)) + (int64_t)16384)])), 512, static_cast<asc_load_l2_cache_mode>(0), 128, min(64, max(((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)) - 64), 0)), 0, 0);
            }
            asc_fill_l1((__cbuf__ uint16_t*)((__cbuf__ uint8_t*)((__cbuf__ bfloat16_t*)buf_dyn_l1 + 122880) + (((min(64, max(((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)) - 64), 0)) + 15) >> 4) * 4096)), (uint32_t)((uint32_t)0), { .repeat = static_cast<uint64_t>((4 - ((min(64, max(((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)) - 64), 0)) + 15) >> 4))), .blk_num = static_cast<uint64_t>(128), .dst_gap = static_cast<uint64_t>(0)});
            asc_copy_l12l0a(((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cbuf__ bfloat16_t*)buf_dyn_l1), 0, 0, 6, 12, 6, 6);
            asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(1));
            asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b + 18432), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 73728), 0, 0, 4, 12, 4, 4);
          }
          asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(1));
          if (__cond_2) {
            asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(2));
          }
          asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>(5));
          asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(1));
          if (__cond_2) {
            asc_mmad(((__cc__ float*)buf_dyn_l0c + 43008), ((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cb__ bfloat16_t*)buf_dyn_l0b + 18432), 96, 192, 64, 0, 1, 0, (bool)1);
          }
          asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(1));
          asc_sync_notify(PIPE_M, PIPE_FIX, static_cast<event_t>(2));
          asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
          asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
          if (__cond_2) {
            asc_copy_l12l0a(((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 36864), 0, 0, 6, 8, 6, 6);
            asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));
          }
          asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(1));
          if (__cond_2) {
            asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b + 18432), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 114688), 0, 0, 4, 8, 4, 4);
          }
          asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(1));
          if (__cond_2) {
            asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(1));
          }
          asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>(4));
          asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(1));
          if (__cond_2) {
            asc_mmad(((__cc__ float*)buf_dyn_l0c + 49152), ((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cb__ bfloat16_t*)buf_dyn_l0b + 18432), 96, 128, 64, 0, 1, 0, (bool)1);
          }
          asc_sync_notify(PIPE_M, PIPE_FIX, static_cast<event_t>(1));
          asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
          asc_sync_intra_wait(PIPE_FIX, 24);
          asc_sync_intra_wait(PIPE_FIX, 8);
          asc_sync_wait(PIPE_M, PIPE_FIX, static_cast<event_t>(2));
          if (__cond_2) {
            asc_set_l0c_copy_nz_para(1, 0, 0);
            asc_copy_l0c2ub((&(((__ubuf__ float*)buf_dyn_shmem)[19968])), ((__cc__ float*)buf_dyn_l0c + 43008), 64, 96, 64, 96, 0, static_cast<asc_dual_dst_mode>(1), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
          }
          asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>(2));
          asc_sync_wait(PIPE_M, PIPE_FIX, static_cast<event_t>(1));
          if (__cond_2) {
            asc_set_l0c_copy_nz_para(1, 0, 0);
            asc_copy_l0c2ub((&(((__ubuf__ float*)buf_dyn_shmem)[23040])), ((__cc__ float*)buf_dyn_l0c + 49152), 64, 96, 64, 96, 0, static_cast<asc_dual_dst_mode>(1), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
          }
          asc_sync_intra_arrive(PIPE_FIX, 3);
          asc_sync_intra_arrive(PIPE_FIX, 19);
          asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>(1));
          asc_sync_intra_wait(PIPE_MTE1, 23);
          asc_sync_intra_wait(PIPE_MTE1, 7);
          asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
          if (__cond_2) {
            asc_copy_l12l0a(((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 131072), 0, 0, 6, 4, 6, 6);
          }
          asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(2));
          if (__cond_2) {
            asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 106496), 0, 0, 8, 4, 8, 8);
          }
          asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
          asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
          if (__cond_2) {
            asc_mmad(((__cc__ float*)buf_dyn_l0c + 30720), ((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cb__ bfloat16_t*)buf_dyn_l0b), 96, 64, 128, 0, 1, 0, (((i * 2) + max(0, (((n0 + len_q) - len_kv) >> 6))) == max(0, (((n0 + len_q) - len_kv) >> 6))));
          }
          asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
          asc_sync_intra_wait(PIPE_MTE1, 22);
          asc_sync_intra_wait(PIPE_MTE1, 6);
          asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
          if (__cond_2) {
            asc_copy_l12l0a(((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 137216), 0, 0, 6, 4, 6, 6);
            asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 61440), 0, 0, 12, 4, 12, 12);
          }
          asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
          asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(0));
          asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
          if (__cond_2) {
            asc_mmad(((__cc__ float*)buf_dyn_l0c), ((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cb__ bfloat16_t*)buf_dyn_l0b), 96, 64, 192, 0, 1, 0, (((i * 2) + max(0, (((n0 + len_q) - len_kv) >> 6))) == max(0, (((n0 + len_q) - len_kv) >> 6))));
          }
          asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
          if (__cond_2) {
            asc_copy_l12l0a_transpose(((__ca__ bfloat16_t*)buf_dyn_l0a + 18432), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 137216), 0, 0, 6, 4, 6, 4);
          }
          asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
          if (__cond_2) {
            asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 18432), 0, 0, 12, 6, 12, 12);
          }
          asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
          asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>(3));
          asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
          if (__cond_2) {
            asc_mmad(((__cc__ float*)buf_dyn_l0c + 18432), ((__ca__ bfloat16_t*)buf_dyn_l0a + 18432), ((__cb__ bfloat16_t*)buf_dyn_l0b), 64, 96, 192, 0, 1, 0, (bool)1);
          }
          asc_sync_notify(PIPE_M, PIPE_FIX, static_cast<event_t>(0));
          if (__cond_2) {
            asc_set_atomic_add_float();
          }
          asc_sync_wait(PIPE_M, PIPE_FIX, static_cast<event_t>(0));
          if (__cond_2) {
            if (0 < (((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128))) {
              asc_set_l0c_copy_nz_para(1, 0, 0);
              asc_copy_l0c2gm((&(dq_acc[((((((int64_t)i) * (int64_t)49152) + (max((int64_t)0, (((((int64_t)n0) + ((int64_t)len_q)) - ((int64_t)len_kv)) >> (int64_t)6)) * (int64_t)24576)) + (((int64_t)q_start) * (int64_t)384)) + (((int64_t)h) * (int64_t)192))])), ((__cc__ float*)buf_dyn_l0c + 18432), 192, min(64, max((((total_q - q_start) - (max(0, (((n0 + len_q) - len_kv) >> 6)) * 64)) - (i * 128)), 0)), 384, 64, static_cast<asc_store_l2_cache_mode>(0), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
            }
          }
          asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>(0));
          if (__cond_2) {
            asc_set_atomic_none();
          }
        }
      }
    }
    asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(5));
    asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(7));
    asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>(6));
    int8_t __cond_3 = ((int8_t)(bool)0);
    if (h < 2) {
      if ((bool)__cond_5) {
        __cond_3 = ((int8_t)((max((((len_q + 63) >> 6) - max(0, (((n0 + len_q) - len_kv) >> 6))), 1) & 1) == 1));
      }
    }
    asc_sync_intra_wait(PIPE_MTE1, 27);
    asc_sync_intra_wait(PIPE_MTE1, 11);
    asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(7));
    asc_sync_pipe(PIPE_MTE1);
    if (h < 2) {
      if ((bool)__cond_5) {
        if ((bool)__cond_3) {
          asc_copy_l12l0a(((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 131072), 0, 0, 6, 4, 6, 6);
        }
      }
    }
    asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(5));
    if (h < 2) {
      if ((bool)__cond_5) {
        if ((bool)__cond_3) {
          asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 106496), 0, 0, 8, 4, 8, 8);
        }
      }
    }
    asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(7));
    asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(7));
    if (h < 2) {
      if ((bool)__cond_5) {
        if ((bool)__cond_3) {
          asc_mmad(((__cc__ float*)buf_dyn_l0c + 30720), ((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cb__ bfloat16_t*)buf_dyn_l0b), 96, 64, 128, 0, 1, 0, (max((((len_q + 63) >> 6) - 1), max(0, (((n0 + len_q) - len_kv) >> 6))) == max(0, (((n0 + len_q) - len_kv) >> 6))));
        }
      }
    }
    asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(6));
    asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(6));
    if (h < 2) {
      if ((bool)__cond_5) {
        if ((bool)__cond_3) {
          asc_copy_l12l0a(((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 137216), 0, 0, 6, 4, 6, 6);
        }
      }
      if ((bool)__cond_5) {
        if ((bool)__cond_3) {
          asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 61440), 0, 0, 12, 4, 12, 12);
        }
      }
    }
    asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(6));
    asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(6));
    if (h < 2) {
      if ((bool)__cond_5) {
        if ((bool)__cond_3) {
          asc_mmad(((__cc__ float*)buf_dyn_l0c), ((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cb__ bfloat16_t*)buf_dyn_l0b), 96, 64, 192, 0, 1, 0, (max((((len_q + 63) >> 6) - 1), max(0, (((n0 + len_q) - len_kv) >> 6))) == max(0, (((n0 + len_q) - len_kv) >> 6))));
        }
      }
    }
    asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(5));
    if (h < 2) {
      if ((bool)__cond_5) {
        if ((bool)__cond_3) {
          asc_copy_l12l0a_transpose(((__ca__ bfloat16_t*)buf_dyn_l0a + 18432), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 137216), 0, 0, 6, 4, 6, 4);
        }
      }
    }
    asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(5));
    if (h < 2) {
      if ((bool)__cond_5) {
        if ((bool)__cond_3) {
          asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 18432), 0, 0, 12, 6, 12, 12);
        }
      }
    }
    asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(5));
    asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>(6));
    asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(5));
    if (h < 2) {
      if ((bool)__cond_5) {
        if ((bool)__cond_3) {
          asc_mmad(((__cc__ float*)buf_dyn_l0c + 18432), ((__ca__ bfloat16_t*)buf_dyn_l0a + 18432), ((__cb__ bfloat16_t*)buf_dyn_l0b), 64, 96, 192, 0, 1, 0, (bool)1);
        }
      }
    }
    asc_sync_notify(PIPE_M, PIPE_FIX, static_cast<event_t>(7));
    if (h < 2) {
      if ((bool)__cond_5) {
        if ((bool)__cond_3) {
          asc_set_atomic_add_float();
        }
      }
    }
    asc_sync_wait(PIPE_M, PIPE_FIX, static_cast<event_t>(7));
    if (h < 2) {
      if ((bool)__cond_5) {
        if ((bool)__cond_3) {
          if (0 < ((total_q - q_start) - (max((((len_q + 63) >> 6) - 1), max(0, (((n0 + len_q) - len_kv) >> 6))) * 64))) {
            asc_set_l0c_copy_nz_para(1, 0, 0);
            asc_copy_l0c2gm((&(dq_acc[(((max((((((int64_t)len_q) + (int64_t)63) >> (int64_t)6) - (int64_t)1), max((int64_t)0, (((((int64_t)n0) + ((int64_t)len_q)) - ((int64_t)len_kv)) >> (int64_t)6))) * (int64_t)24576) + (((int64_t)q_start) * (int64_t)384)) + (((int64_t)h) * (int64_t)192))])), ((__cc__ float*)buf_dyn_l0c + 18432), 192, min(64, max(((total_q - q_start) - (max((((len_q + 63) >> 6) - 1), max(0, (((n0 + len_q) - len_kv) >> 6))) * 64)), 0)), 384, 64, static_cast<asc_store_l2_cache_mode>(0), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
          }
        }
      }
      if ((bool)__cond_5) {
        if ((bool)__cond_3) {
          asc_set_atomic_none();
        }
      }
      if ((bool)__cond_5) {
        if (!((bool)__cond_3)) {
          asc_copy_l12l0a(((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 143360), 0, 0, 6, 4, 6, 6);
        }
      }
      if ((bool)__cond_5) {
        if (!((bool)__cond_3)) {
          asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 122880), 0, 0, 8, 4, 8, 8);
        }
      }
    }
    asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(4));
    asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(4));
    if (h < 2) {
      if ((bool)__cond_5) {
        if (!((bool)__cond_3)) {
          asc_mmad(((__cc__ float*)buf_dyn_l0c + 30720), ((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cb__ bfloat16_t*)buf_dyn_l0b), 96, 64, 128, 0, 1, 0, (max((((len_q + 63) >> 6) - 1), max(0, (((n0 + len_q) - len_kv) >> 6))) == max(0, (((n0 + len_q) - len_kv) >> 6))));
        }
      }
    }
    asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(4));
    asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(4));
    if (h < 2) {
      if ((bool)__cond_5) {
        if (!((bool)__cond_3)) {
          asc_copy_l12l0a(((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 149504), 0, 0, 6, 4, 6, 6);
        }
      }
      if ((bool)__cond_5) {
        if (!((bool)__cond_3)) {
          asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 86016), 0, 0, 12, 4, 12, 12);
        }
      }
    }
    asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(3));
    asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(3));
    if (h < 2) {
      if ((bool)__cond_5) {
        if (!((bool)__cond_3)) {
          asc_mmad(((__cc__ float*)buf_dyn_l0c), ((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cb__ bfloat16_t*)buf_dyn_l0b), 96, 64, 192, 0, 1, 0, (max((((len_q + 63) >> 6) - 1), max(0, (((n0 + len_q) - len_kv) >> 6))) == max(0, (((n0 + len_q) - len_kv) >> 6))));
        }
      }
    }
    asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(3));
    if (h < 2) {
      if ((bool)__cond_5) {
        if (!((bool)__cond_3)) {
          asc_copy_l12l0a_transpose(((__ca__ bfloat16_t*)buf_dyn_l0a + 18432), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 149504), 0, 0, 6, 4, 6, 4);
        }
      }
    }
    asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(3));
    if (h < 2) {
      if ((bool)__cond_5) {
        if (!((bool)__cond_3)) {
          asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 18432), 0, 0, 12, 6, 12, 12);
        }
      }
    }
    asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(2));
    asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(2));
    if (h < 2) {
      if ((bool)__cond_5) {
        if (!((bool)__cond_3)) {
          asc_mmad(((__cc__ float*)buf_dyn_l0c + 18432), ((__ca__ bfloat16_t*)buf_dyn_l0a + 18432), ((__cb__ bfloat16_t*)buf_dyn_l0b), 64, 96, 192, 0, 1, 0, (bool)1);
        }
      }
    }
    asc_sync_notify(PIPE_M, PIPE_FIX, static_cast<event_t>(6));
    if (h < 2) {
      if ((bool)__cond_5) {
        if (!((bool)__cond_3)) {
          asc_set_atomic_add_float();
        }
      }
    }
    asc_sync_wait(PIPE_M, PIPE_FIX, static_cast<event_t>(6));
    if (h < 2) {
      if ((bool)__cond_5) {
        if (!((bool)__cond_3)) {
          if (0 < ((total_q - q_start) - (max((((len_q + 63) >> 6) - 1), max(0, (((n0 + len_q) - len_kv) >> 6))) * 64))) {
            asc_set_l0c_copy_nz_para(1, 0, 0);
            asc_copy_l0c2gm((&(dq_acc[(((max((((((int64_t)len_q) + (int64_t)63) >> (int64_t)6) - (int64_t)1), max((int64_t)0, (((((int64_t)n0) + ((int64_t)len_q)) - ((int64_t)len_kv)) >> (int64_t)6))) * (int64_t)24576) + (((int64_t)q_start) * (int64_t)384)) + (((int64_t)h) * (int64_t)192))])), ((__cc__ float*)buf_dyn_l0c + 18432), 192, min(64, max(((total_q - q_start) - (max((((len_q + 63) >> 6) - 1), max(0, (((n0 + len_q) - len_kv) >> 6))) * 64)), 0)), 384, 64, static_cast<asc_store_l2_cache_mode>(0), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
          }
        }
      }
      if ((bool)__cond_5) {
        if (!((bool)__cond_3)) {
          asc_set_atomic_none();
        }
      }
      if ((bool)__cond_5) {
        asc_set_l0c_copy_nz_para(1, 0, 0);
        asc_copy_l0c2ub((&(((__ubuf__ float*)buf_dyn_shmem)[0])), ((__cc__ float*)buf_dyn_l0c), 192, 96, 192, 96, 0, static_cast<asc_dual_dst_mode>(1), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
      }
      if ((bool)__cond_5) {
        asc_set_l0c_copy_nz_para(1, 0, 0);
        asc_copy_l0c2ub((&(((__ubuf__ float*)buf_dyn_shmem)[9216])), ((__cc__ float*)buf_dyn_l0c + 30720), 128, 96, 128, 96, 0, static_cast<asc_dual_dst_mode>(1), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
      }
    }
    asc_sync_intra_arrive(PIPE_FIX, 10);
    asc_sync_intra_arrive(PIPE_FIX, 26);
    asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>(0));
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(0));
    asc_sync_intra_wait(PIPE_MTE1, 0);
    asc_sync_intra_wait(PIPE_MTE1, 16);
    asc_sync_intra_wait(PIPE_MTE1, 1);
    asc_sync_intra_wait(PIPE_MTE1, 17);
    asc_sync_intra_wait(PIPE_FIX, 2);
    asc_sync_intra_wait(PIPE_FIX, 18);
    asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>(1));
    asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>(2));
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(1));
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(2));
  }
  if ASC_IS_AIV {
    int32_t sid = asc_get_sub_block_id();
    asc_sync_intra_arrive(PIPE_MTE3, 0);
    asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
    asc_sync_intra_arrive(PIPE_MTE3, 1);
    asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
    asc_sync_intra_arrive(PIPE_V, 2);
    asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
    asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
    int32_t h_1 = (h_off[0] + (((int32_t)block_idx) & 1));
    int64_t sid_1 = asc_get_sub_block_id();
    if (h_1 < 2) {
      cnt[0] = 0;
      req[0] = 0;
      n0_l[0] = 0;
      for (int32_t g_1 = 0; g_1 < 2; ++g_1) {
        int32_t nt_1 = ((((cu_kv[(g_1 + 1)] + 95) - cu_kv[g_1]) / 96) + ((((cu_kv[(g_1 + 1)] + 95) - cu_kv[g_1]) % 96) >> 31));
        bool __cond_0_1 = ((cnt[0] <= (((int32_t)block_idx) >> 1)) & ((((int32_t)block_idx) >> 1) < (cnt[0] + nt_1)));
        if (__cond_0_1) {
          req[0] = g_1;
          n0_l[0] = (((((int32_t)block_idx) >> 1) * 96) - (cnt[0] * 96));
        }
        cnt[0] = (cnt[0] + nt_1);
      }
      int32_t condval_5;
      if (((((int32_t)block_idx) >> 1) < cnt[0])) {
        condval_5 = 1;
      } else {
        condval_5 = 0;
      }
      live[0] = condval_5;
    }
    int8_t __cond_5_1 = ((int8_t)(bool)0);
    if (h_1 < 2) {
      __cond_5_1 = ((int8_t)(live[0] == 1));
    }
    int32_t n0_1 = 0;
    if (h_1 < 2) {
      if ((bool)__cond_5_1) {
        n0_1 = n0_l[0];
      }
    }
    int32_t q_start_1 = 0;
    if (h_1 < 2) {
      if ((bool)__cond_5_1) {
        int32_t condval_6;
        if (((0 <= req[0]) && (req[0] < 3))) {
          condval_6 = cu_q[req[(int64_t)0]];
        } else {
          condval_6 = 0;
        }
        q_start_1 = condval_6;
      }
    }
    int32_t kv_start_1 = 0;
    if (h_1 < 2) {
      if ((bool)__cond_5_1) {
        int32_t condval_7;
        if (((0 <= req[0]) && (req[0] < 3))) {
          condval_7 = cu_kv[req[(int64_t)0]];
        } else {
          condval_7 = 0;
        }
        kv_start_1 = condval_7;
      }
    }
    int32_t len_q_1 = 0;
    if (h_1 < 2) {
      if ((bool)__cond_5_1) {
        int32_t condval_8;
        if (((-1 <= req[0]) && (req[0] < 2))) {
          condval_8 = cu_q[(((int64_t)req[(int64_t)0]) + (int64_t)1)];
        } else {
          condval_8 = 0;
        }
        len_q_1 = (condval_8 - q_start_1);
      }
    }
    int32_t len_kv_1 = 0;
    if (h_1 < 2) {
      if ((bool)__cond_5_1) {
        int32_t condval_9;
        if (((-1 <= req[0]) && (req[0] < 2))) {
          condval_9 = cu_kv[(((int64_t)req[(int64_t)0]) + (int64_t)1)];
        } else {
          condval_9 = 0;
        }
        len_kv_1 = (condval_9 - kv_start_1);
      }
      if ((bool)__cond_5_1) {
        for (int32_t i_1 = 0; i_1 < ((max((((len_q_1 + 63) >> 6) - max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6))), 1) + 1) >> 1); ++i_1) {
          asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
          if (0 < (((total_q - q_start_1) - (max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6)) * 64)) - (i_1 * 128))) {
            asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[38528]))), (__gm__ uint8_t*)((&(lse[((((((int64_t)i_1) * (int64_t)128) + (max((int64_t)0, (((((int64_t)n0_1) + ((int64_t)len_q_1)) - ((int64_t)len_kv_1)) >> (int64_t)6)) * (int64_t)64)) + (((int64_t)h_1) * ((int64_t)total_q))) + ((int64_t)q_start_1))]))), 1, (min(64, max((((total_q - q_start_1) - (max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6)) * 64)) - (i_1 * 128)), 0)) * 4), 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), (min(64, max((((total_q - q_start_1) - (max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6)) * 64)) - (i_1 * 128)), 0)) * 4), (min(64, max((((total_q - q_start_1) - (max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6)) * 64)) - (i_1 * 128)), 0)) * 4));
          }
          if (0 < (((total_q - q_start_1) - (max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6)) * 64)) - (i_1 * 128))) {
            asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[38592]))), (__gm__ uint8_t*)((&(delta[((((((int64_t)i_1) * (int64_t)128) + (max((int64_t)0, (((((int64_t)n0_1) + ((int64_t)len_q_1)) - ((int64_t)len_kv_1)) >> (int64_t)6)) * (int64_t)64)) + (((int64_t)h_1) * ((int64_t)total_q))) + ((int64_t)q_start_1))]))), 1, (min(64, max((((total_q - q_start_1) - (max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6)) * 64)) - (i_1 * 128)), 0)) * 4), 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), (min(64, max((((total_q - q_start_1) - (max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6)) * 64)) - (i_1 * 128)), 0)) * 4), (min(64, max((((total_q - q_start_1) - (max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6)) * 64)) - (i_1 * 128)), 0)) * 4));
          }
          asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(1));
          bool fast = (((n0_1 + 96) <= len_kv_1) & ((((i_1 * 128) + (max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6)) * 64)) + 64) <= len_q_1));
          bool fast_1 = (fast & ((n0_1 + 95) <= ((((i_1 * 128) + (max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6)) * 64)) + len_kv_1) - len_q_1)));
          asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(1));
          asc_sync_intra_wait(PIPE_V, 9);
          if (fast_1) {
            asc_vf_call<_bwd_main_kernel_simt_vf_0>(cce::dim3(128), buf_dyn_shmem);
          }
          if (!fast_1) {
            asc_vf_call<_bwd_main_kernel_simt_vf_1>(cce::dim3(128), sid_1, n0_1, len_kv_1, i_1, len_q_1, buf_dyn_shmem);
          }
          asc_sync_intra_arrive(PIPE_V, 8);
          asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
          ascend_nd2nz_scatter<48, 64, bfloat16_t, bfloat16_t>((__ubuf__ bfloat16_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[70912]))), (__ubuf__ bfloat16_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[58368]))));
          asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(3));
          asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(3));
          asc_copy_ub2l1((__cbuf__ void*)(((__cbuf__ bfloat16_t*)buf_dyn_l1 + ((sid * 768) + 131072))), (__ubuf__ void*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[58368]))), 4, 48, 1, 48);
          asc_sync_intra_arrive(PIPE_MTE3, 7);
          ascend_nd2nz_scatter<48, 64, bfloat16_t, bfloat16_t>((__ubuf__ bfloat16_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[73984]))), (__ubuf__ bfloat16_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[61504]))));
          asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(2));
          asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(2));
          asc_copy_ub2l1((__cbuf__ void*)(((__cbuf__ bfloat16_t*)buf_dyn_l1 + ((sid * 768) + 137216))), (__ubuf__ void*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[61504]))), 4, 48, 1, 48);
          asc_sync_intra_arrive(PIPE_MTE3, 6);
          if (0 < i_1) {
            asc_set_atomic_add_float();
            asc_set_atomic_none();
          }
          bool __cond_2_1 = ((((i_1 * 2) + max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6))) + 1) < ((len_q_1 + 63) >> 6));
          if (__cond_2_1) {
            asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
            if (64 < (((total_q - q_start_1) - (max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6)) * 64)) - (i_1 * 128))) {
              asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[38656]))), (__gm__ uint8_t*)((&(lse[(((((((int64_t)i_1) * (int64_t)128) + (max((int64_t)0, (((((int64_t)n0_1) + ((int64_t)len_q_1)) - ((int64_t)len_kv_1)) >> (int64_t)6)) * (int64_t)64)) + (((int64_t)h_1) * ((int64_t)total_q))) + ((int64_t)q_start_1)) + (int64_t)64)]))), 1, (min(64, max(((((total_q - q_start_1) - (max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6)) * 64)) - (i_1 * 128)) - 64), 0)) * 4), 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), (min(64, max(((((total_q - q_start_1) - (max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6)) * 64)) - (i_1 * 128)) - 64), 0)) * 4), (min(64, max(((((total_q - q_start_1) - (max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6)) * 64)) - (i_1 * 128)) - 64), 0)) * 4));
            }
            if (64 < (((total_q - q_start_1) - (max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6)) * 64)) - (i_1 * 128))) {
              asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[38720]))), (__gm__ uint8_t*)((&(delta[(((((((int64_t)i_1) * (int64_t)128) + (max((int64_t)0, (((((int64_t)n0_1) + ((int64_t)len_q_1)) - ((int64_t)len_kv_1)) >> (int64_t)6)) * (int64_t)64)) + (((int64_t)h_1) * ((int64_t)total_q))) + ((int64_t)q_start_1)) + (int64_t)64)]))), 1, (min(64, max(((((total_q - q_start_1) - (max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6)) * 64)) - (i_1 * 128)) - 64), 0)) * 4), 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), (min(64, max(((((total_q - q_start_1) - (max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6)) * 64)) - (i_1 * 128)) - 64), 0)) * 4), (min(64, max(((((total_q - q_start_1) - (max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6)) * 64)) - (i_1 * 128)) - 64), 0)) * 4));
            }
            asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
          }
          int8_t fast_2 = ((int8_t)(bool)0);
          if (__cond_2_1) {
            fast_2 = ((int8_t)(((n0_1 + 96) <= len_kv_1) & ((((i_1 * 128) + (max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6)) * 64)) + 128) <= len_q_1)));
          }
          int8_t fast_3 = ((int8_t)(bool)0);
          if (__cond_2_1) {
            fast_3 = ((int8_t)(((bool)fast_2) & ((n0_1 + 31) <= ((((i_1 * 128) + (max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6)) * 64)) + len_kv_1) - len_q_1))));
            asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
          }
          asc_sync_intra_wait(PIPE_V, 3);
          if (__cond_2_1) {
            if ((bool)fast_3) {
              asc_vf_call<_bwd_main_kernel_simt_vf_2>(cce::dim3(128), buf_dyn_shmem);
            }
            if (!((bool)fast_3)) {
              asc_vf_call<_bwd_main_kernel_simt_vf_3>(cce::dim3(128), sid_1, n0_1, len_kv_1, i_1, len_q_1, buf_dyn_shmem);
            }
          }
          asc_sync_intra_arrive(PIPE_V, 2);
          if (__cond_2_1) {
            asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
            asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
            ascend_nd2nz_scatter<48, 64, bfloat16_t, bfloat16_t>((__ubuf__ bfloat16_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[70912]))), (__ubuf__ bfloat16_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[64640]))));
            asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(1));
          }
          asc_sync_intra_wait(PIPE_MTE3, 5);
          if (__cond_2_1) {
            asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(1));
            asc_copy_ub2l1((__cbuf__ void*)(((__cbuf__ bfloat16_t*)buf_dyn_l1 + ((sid * 768) + 143360))), (__ubuf__ void*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[64640]))), 4, 48, 1, 48);
          }
          asc_sync_intra_arrive(PIPE_MTE3, 1);
          if (__cond_2_1) {
            asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
            asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
            ascend_nd2nz_scatter<48, 64, bfloat16_t, bfloat16_t>((__ubuf__ bfloat16_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[73984]))), (__ubuf__ bfloat16_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[67776]))));
            asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
          }
          asc_sync_intra_wait(PIPE_MTE3, 4);
          if (__cond_2_1) {
            asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
            asc_copy_ub2l1((__cbuf__ void*)(((__cbuf__ bfloat16_t*)buf_dyn_l1 + ((sid * 768) + 149504))), (__ubuf__ void*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[67776]))), 4, 48, 1, 48);
          }
          asc_sync_intra_arrive(PIPE_MTE3, 0);
          if (__cond_2_1) {
            asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
            asc_set_atomic_add_float();
            asc_set_atomic_none();
          }
        }
      }
    }
    asc_sync_intra_arrive(PIPE_MTE3, 11);
    int8_t __cond_3_1 = ((int8_t)(bool)0);
    if (h_1 < 2) {
      if ((bool)__cond_5_1) {
        __cond_3_1 = ((int8_t)((max((((len_q_1 + 63) >> 6) - max(0, (((n0_1 + len_q_1) - len_kv_1) >> 6))), 1) & 1) == 1));
      }
      if ((bool)__cond_5_1) {
        if ((bool)__cond_3_1) {
          asc_set_atomic_add_float();
        }
      }
      if ((bool)__cond_5_1) {
        if ((bool)__cond_3_1) {
          asc_set_atomic_none();
        }
      }
      if ((bool)__cond_5_1) {
        if (!((bool)__cond_3_1)) {
          asc_set_atomic_add_float();
        }
      }
      if ((bool)__cond_5_1) {
        if (!((bool)__cond_3_1)) {
          asc_set_atomic_none();
        }
      }
    }
    asc_sync_intra_wait(PIPE_V, 10);
    if (h_1 < 2) {
      if ((bool)__cond_5_1) {
        asc_vf_call<_bwd_main_kernel_simt_vf_4>(cce::dim3(128), buf_dyn_shmem);
      }
    }
    asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(4));
    int8_t __cond_4 = ((int8_t)(bool)0);
    if (h_1 < 2) {
      if ((bool)__cond_5_1) {
        __cond_4 = ((int8_t)(((((((int32_t)sid_1) * 48) + kv_start_1) + n0_1) + 48) <= (kv_start_1 + len_kv_1)));
      }
    }
    asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(4));
    if (h_1 < 2) {
      if ((bool)__cond_5_1) {
        if ((bool)__cond_4) {
          if (0 < (((total_kv - kv_start_1) - n0_1) - (((int32_t)sid_1) * 48))) {
            asc_copy_ub2gm_align((__gm__ uint8_t*)((&(dk[((((sid_1 * (int64_t)18432) + (((int64_t)kv_start_1) * (int64_t)384)) + (((int64_t)n0_1) * (int64_t)384)) + (((int64_t)h_1) * (int64_t)192))]))), (__ubuf__ uint8_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[30720]))), min(48, max((((total_kv - kv_start_1) - n0_1) - (((int32_t)sid_1) * 48)), 0)), 384, static_cast<asc_store_l2_cache_mode>(4), 768, 384);
          }
        }
      }
      if ((bool)__cond_5_1) {
        if ((bool)__cond_4) {
          if (0 < (((total_kv - kv_start_1) - n0_1) - (((int32_t)sid_1) * 48))) {
            asc_copy_ub2gm_align((__gm__ uint8_t*)((&(dv[((((sid_1 * (int64_t)12288) + (((int64_t)kv_start_1) * (int64_t)256)) + (((int64_t)n0_1) * (int64_t)256)) + (((int64_t)h_1) * (int64_t)128))]))), (__ubuf__ uint8_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[52224]))), min(48, max((((total_kv - kv_start_1) - n0_1) - (((int32_t)sid_1) * 48)), 0)), 256, static_cast<asc_store_l2_cache_mode>(4), 512, 256);
          }
        }
      }
      if ((bool)__cond_5_1) {
        if (!((bool)__cond_4)) {
          asc_vf_call<_bwd_main_kernel_simt_vf_5>(cce::dim3(128), h_1, sid_1, kv_start_1, n0_1, len_kv_1, total_kv, dk, buf_dyn_shmem, dv);
        }
      }
    }
    asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
    asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
    asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
    asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
  }
}

