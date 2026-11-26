#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/nd2nz_copy.h>
__simt_vf__ __launch_bounds__(128) inline void _bwd_main_kernel_simt_vf_1(__ubuf__ uint8_t* buf_dyn_shmem, int64_t sid_1, int32_t kq_1) {
  #pragma unroll
  for (int32_t i = 0; i < 24; ++i) {
    float p = expf(((((__ubuf__ float*)buf_dyn_shmem)[((i * 128) + ((int32_t)threadIdx.x))] * float(0x1.279a74590331cp-4f/*7.216878e-02*/)) - ((__ubuf__ float*)buf_dyn_shmem)[((((int32_t)threadIdx.x) & 63) + 15360)]));
    float condval;
    if (((((((((int32_t)block_idx) % 6) * 3) + (((((int32_t)sid_1) * 3) + (i >> 3)) >> 1)) < 16) & (bool)1) & ((((((((int32_t)block_idx) % 6) * 96) + (((int32_t)sid_1) * 48)) + (i * 2)) + (((int32_t)threadIdx.x) >> 6)) <= ((kq_1 * 64) + (((int32_t)threadIdx.x) & 63))))) {
      condval = p;
    } else {
      condval = float(0x0p+0f/*0.000000e+00*/);
    }
    ((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((i * 128) + ((int32_t)threadIdx.x)) + 52352)] = ((bfloat16_t)condval);
    float condval_1;
    if (((((((((int32_t)block_idx) % 6) * 3) + (((((int32_t)sid_1) * 3) + (i >> 3)) >> 1)) < 16) & (bool)1) & ((((((((int32_t)block_idx) % 6) * 96) + (((int32_t)sid_1) * 48)) + (i * 2)) + (((int32_t)threadIdx.x) >> 6)) <= ((kq_1 * 64) + (((int32_t)threadIdx.x) & 63))))) {
      condval_1 = p;
    } else {
      condval_1 = float(0x0p+0f/*0.000000e+00*/);
    }
    ((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((i * 128) + ((int32_t)threadIdx.x)) + 55424)] = ((bfloat16_t)(condval_1 * (((__ubuf__ float*)buf_dyn_shmem)[(((i * 128) + ((int32_t)threadIdx.x)) + 9216)] - ((__ubuf__ float*)buf_dyn_shmem)[((((int32_t)threadIdx.x) & 63) + 19968)])));
  }
}

__simt_vf__ __launch_bounds__(128) inline void _bwd_main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  #pragma unroll
  for (int32_t i = 0; i < 12; ++i) {
    float broadcast_var = float(0x1.279a74590331cp-4f/*7.216878e-02*/);
    float2 _;
      float2 v_ = (*(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) + ((i * 256) + (((int32_t)threadIdx.x) * 2))) * make_float2(broadcast_var, broadcast_var)) - *(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) + (((((int32_t)threadIdx.x) & 31) * 2) + 15360));
      _.x = expf(v_.x);
      _.y = expf(v_.y);
    float2 p = _;
    uint1 __1;
      (reinterpret_cast<bfloat16x2_t*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&p))[0]);
    *(__ubuf__ uint1*)(((__ubuf__ bfloat16_t*)buf_dyn_shmem) + (((i * 256) + (((int32_t)threadIdx.x) * 2)) + 52352)) = __1;
    uint1 __2;
      float2 v__1 = p * (*(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) + (((i * 256) + (((int32_t)threadIdx.x) * 2)) + 9216)) - *(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) + (((((int32_t)threadIdx.x) & 31) * 2) + 19968)));
      (reinterpret_cast<bfloat16x2_t*>(&__2))[0] = __float22bfloat162_rn(((float2*)(&v__1))[0]);
    *(__ubuf__ uint1*)(((__ubuf__ bfloat16_t*)buf_dyn_shmem) + (((i * 256) + (((int32_t)threadIdx.x) * 2)) + 55424)) = __2;
  }
}

__simt_vf__ __launch_bounds__(128) inline void _bwd_main_kernel_simt_vf_2(__ubuf__ uint8_t* buf_dyn_shmem) {
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
    *(__ubuf__ uint1*)(((__ubuf__ bfloat16_t*)buf_dyn_shmem) + (((i_1 * 256) + (((int32_t)threadIdx.x) * 2)) + 39936)) = *(uint1*)(out_v_local_cast_2 + 0);
  }
}

extern "C" __global__ __mix__(1, 2) void _bwd_main_kernel(__gm__ float* delta, __gm__ bfloat16_t* dk, __gm__ bfloat16_t* dout, __gm__ float* dq_acc, __gm__ bfloat16_t* dv, __gm__ bfloat16_t* k, __gm__ float* lse, __gm__ bfloat16_t* q, __gm__ bfloat16_t* v) {
  asc_init();
  __cbuf__ uint8_t *buf_dyn_l1 = (__cbuf__ uint8_t *)0;
  __cc__ uint8_t *buf_dyn_l0c = (__cc__ uint8_t *)0;
  __cb__ uint8_t *buf_dyn_l0b = (__cb__ uint8_t *)0;
  __ca__ uint8_t *buf_dyn_l0a = (__ca__ uint8_t *)0;
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  if ASC_IS_AIC {
    asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(0));
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(1));
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(2));
    asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(96), 0);
    asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + 24576), (&(k[(((((int32_t)block_idx) % 6) * 36864) + ((((int32_t)block_idx) / 6) * 192))])), 768, static_cast<asc_load_l2_cache_mode>(0), min(96, (512 - ((((int32_t)block_idx) % 6) * 96))), 192, 0, 0);
    asc_fill_l1((__cbuf__ uint16_t*)((__cbuf__ uint8_t*)((__cbuf__ bfloat16_t*)buf_dyn_l1 + 24576) + (min(96, (512 - ((((int32_t)block_idx) % 6) * 96))) * 32)), (uint32_t)((uint32_t)0), { .repeat = static_cast<uint64_t>(12), .blk_num = static_cast<uint64_t>(max(0, (((((int32_t)block_idx) % 6) * 96) - 416))), .dst_gap = static_cast<uint64_t>((96 - max(0, (((((int32_t)block_idx) % 6) * 96) - 416))))});
    asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(3));
    asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(96), 0);
    asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + 61440), (&(v[(((((int32_t)block_idx) % 6) * 24576) + ((((int32_t)block_idx) / 6) * 128))])), 512, static_cast<asc_load_l2_cache_mode>(0), min(96, (512 - ((((int32_t)block_idx) % 6) * 96))), 128, 0, 0);
    asc_fill_l1((__cbuf__ uint16_t*)((__cbuf__ uint8_t*)((__cbuf__ bfloat16_t*)buf_dyn_l1 + 61440) + (min(96, (512 - ((((int32_t)block_idx) % 6) * 96))) * 32)), (uint32_t)((uint32_t)0), { .repeat = static_cast<uint64_t>(8), .blk_num = static_cast<uint64_t>(max(0, (((((int32_t)block_idx) % 6) * 96) - 416))), .dst_gap = static_cast<uint64_t>((96 - max(0, (((((int32_t)block_idx) % 6) * 96) - 416))))});
    asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(192), 0);
    asc_copy_gm2l1_dn2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + 43008), (&(k[(((((int32_t)block_idx) % 6) * 36864) + ((((int32_t)block_idx) / 6) * 192))])), 768, static_cast<asc_load_l2_cache_mode>(0), 192, min(96, (512 - ((((int32_t)block_idx) % 6) * 96))), 0, 0);
    asc_fill_l1((__cbuf__ uint16_t*)((__cbuf__ uint8_t*)((__cbuf__ bfloat16_t*)buf_dyn_l1 + 43008) + (((min(96, (512 - ((((int32_t)block_idx) % 6) * 96))) + 15) >> 4) * 6144)), (uint32_t)((uint32_t)0), { .repeat = static_cast<uint64_t>((6 - ((min(96, (512 - ((((int32_t)block_idx) % 6) * 96))) + 15) >> 4))), .blk_num = static_cast<uint64_t>(192), .dst_gap = static_cast<uint64_t>(0)});
    for (int32_t kq = (((((int32_t)block_idx) % 6) * 96) >> 6); kq < 8; ++kq) {
      asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(2));
      asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(64), 0);
      asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + 73728), (&(q[((kq * 24576) + ((((int32_t)block_idx) / 6) * 192))])), 768, static_cast<asc_load_l2_cache_mode>(0), 64, 192, 0, 0);
      asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(2));
      asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(192), 0);
      asc_copy_gm2l1_dn2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + (((kq - (((((int32_t)block_idx) % 6) * 96) >> 6)) & 1) * 12288)), (&(q[((kq * 24576) + ((((int32_t)block_idx) / 6) * 192))])), 768, static_cast<asc_load_l2_cache_mode>(0), 192, 64, 0, 0);
      asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(1));
      asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(128), 0);
      asc_copy_gm2l1_dn2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + 94208), (&(dout[((kq * 16384) + ((((int32_t)block_idx) / 6) * 128))])), 512, static_cast<asc_load_l2_cache_mode>(0), 128, 64, 0, 0);
      asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(1));
      asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(0));
      asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(64), 0);
      asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + 86016), (&(dout[((kq * 16384) + ((((int32_t)block_idx) / 6) * 128))])), 512, static_cast<asc_load_l2_cache_mode>(0), 64, 128, 0, 0);
      asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));
      if (kq == (((((int32_t)block_idx) % 6) * 96) >> 6)) {
        asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(3));
      }
      asc_copy_l12l0a(((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 24576), 0, 0, 6, 12, 6, 6);
      asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(2));
      asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b + 18432), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 73728), 0, 0, 4, 12, 4, 4);
      asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(4));
      asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(2));
      asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
      asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(1));
      asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 94208), 0, 0, 8, 4, 8, 8);
      asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(1));
      asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(4));
      asc_mmad(((__cc__ float*)buf_dyn_l0c + 43008), ((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cb__ bfloat16_t*)buf_dyn_l0b + 18432), 96, 192, 64, 0, 1, 0, (bool)1);
      asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(3));
      asc_sync_notify(PIPE_M, PIPE_FIX, static_cast<event_t>(2));
      asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));
      asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(3));
      asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b + 18432), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 86016), 0, 0, 4, 8, 4, 4);
      asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(0));
      asc_copy_l12l0a(((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 61440), 0, 0, 6, 8, 6, 6);
      asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(3));
      asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(3));
      asc_mmad(((__cc__ float*)buf_dyn_l0c + 49152), ((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cb__ bfloat16_t*)buf_dyn_l0b + 18432), 96, 128, 64, 0, 1, 0, (bool)1);
      asc_sync_notify(PIPE_M, PIPE_FIX, static_cast<event_t>(1));
      asc_sync_wait(PIPE_M, PIPE_FIX, static_cast<event_t>(2));
      asc_set_l0c_copy_nz_para(1, 0, 0);
      asc_copy_l0c2ub((&(((__ubuf__ float*)buf_dyn_shmem)[0])), ((__cc__ float*)buf_dyn_l0c + 43008), 64, 96, 64, 96, 0, static_cast<asc_dual_dst_mode>(1), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
      asc_sync_wait(PIPE_M, PIPE_FIX, static_cast<event_t>(1));
      asc_set_l0c_copy_nz_para(1, 0, 0);
      asc_copy_l0c2ub((&(((__ubuf__ float*)buf_dyn_shmem)[9216])), ((__cc__ float*)buf_dyn_l0c + 49152), 64, 96, 64, 96, 0, static_cast<asc_dual_dst_mode>(1), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
      asc_sync_intra_arrive(PIPE_FIX, 2);
      asc_sync_intra_arrive(PIPE_FIX, 18);
      asc_sync_intra_wait(PIPE_MTE1, 17);
      asc_sync_intra_wait(PIPE_MTE1, 1);
      asc_copy_l12l0a(((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 102400), 0, 0, 6, 4, 6, 6);
      asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(2));
      asc_sync_intra_wait(PIPE_MTE1, 16);
      asc_sync_intra_wait(PIPE_MTE1, 0);
      asc_copy_l12l0a_transpose(((__ca__ bfloat16_t*)buf_dyn_l0a + 18432), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 108544), 0, 0, 6, 4, 6, 4);
      asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(2));
      asc_mmad(((__cc__ float*)buf_dyn_l0c + 30720), ((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cb__ bfloat16_t*)buf_dyn_l0b), 96, 64, 128, 0, 1, 0, (kq == (((((int32_t)block_idx) % 6) * 96) >> 6)));
      asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(2));
      asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(2));
      asc_copy_l12l0a(((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 108544), 0, 0, 6, 4, 6, 6);
      asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + (((kq - (((((int32_t)block_idx) % 6) * 96) >> 6)) & 1) * 12288)), 0, 0, 12, 4, 12, 12);
      asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(1));
      asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(1));
      asc_mmad(((__cc__ float*)buf_dyn_l0c), ((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cb__ bfloat16_t*)buf_dyn_l0b), 96, 64, 192, 0, 1, 0, (kq == (((((int32_t)block_idx) % 6) * 96) >> 6)));
      asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(1));
      asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(1));
      asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 43008), 0, 0, 12, 6, 12, 12);
      asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
      asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
      asc_mmad(((__cc__ float*)buf_dyn_l0c + 18432), ((__ca__ bfloat16_t*)buf_dyn_l0a + 18432), ((__cb__ bfloat16_t*)buf_dyn_l0b), 64, 96, 192, 0, 1, 0, (bool)1);
      asc_sync_notify(PIPE_M, PIPE_FIX, static_cast<event_t>(0));
      asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
      asc_set_atomic_add_float();
      asc_sync_wait(PIPE_M, PIPE_FIX, static_cast<event_t>(0));
      asc_set_l0c_copy_nz_para(1, 0, 0);
      asc_copy_l0c2gm((&(dq_acc[((kq * 24576) + ((((int32_t)block_idx) / 6) * 192))])), ((__cc__ float*)buf_dyn_l0c + 18432), 192, 64, 384, 64, static_cast<asc_store_l2_cache_mode>(0), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
      asc_set_atomic_none();
    }
    asc_set_l0c_copy_nz_para(1, 0, 0);
    asc_copy_l0c2ub((&(((__ubuf__ float*)buf_dyn_shmem)[0])), ((__cc__ float*)buf_dyn_l0c), 192, 96, 192, 96, 0, static_cast<asc_dual_dst_mode>(1), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
    asc_set_l0c_copy_nz_para(1, 0, 0);
    asc_copy_l0c2ub((&(((__ubuf__ float*)buf_dyn_shmem)[9216])), ((__cc__ float*)buf_dyn_l0c + 30720), 128, 96, 128, 96, 0, static_cast<asc_dual_dst_mode>(1), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
    asc_sync_intra_arrive(PIPE_FIX, 3);
    asc_sync_intra_arrive(PIPE_FIX, 19);
    asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(0));
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(1));
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(2));
  }
  if ASC_IS_AIV {
    int32_t sid = asc_get_sub_block_id();
    asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
    int64_t sid_1 = asc_get_sub_block_id();
    for (int32_t kq_1 = (((((int32_t)block_idx) % 6) * 96) >> 6); kq_1 < 8; ++kq_1) {
      asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
      asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[15360]))), (__gm__ uint8_t*)((&(lse[(((((int32_t)block_idx) / 6) * 512) + (kq_1 * 64))]))), 1, 256, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 256, 256);
      asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[19968]))), (__gm__ uint8_t*)((&(delta[(((((int32_t)block_idx) / 6) * 512) + (kq_1 * 64))]))), 1, 256, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 256, 256);
      asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
      asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
      asc_sync_intra_wait(PIPE_V, 2);
      if (!((((((int32_t)block_idx) % 6) <= 4) & (kq_1 <= 7)) & ((((((int32_t)block_idx) % 6) * 96) + 95) < (kq_1 * 64)))) {
        asc_vf_call<_bwd_main_kernel_simt_vf_1>(cce::dim3(128), buf_dyn_shmem, sid_1, kq_1);
      }
      if ((((((int32_t)block_idx) % 6) <= 4) & (kq_1 <= 7)) & ((((((int32_t)block_idx) % 6) * 96) + 95) < (kq_1 * 64))) {
        asc_vf_call<_bwd_main_kernel_simt_vf_0>(cce::dim3(128), buf_dyn_shmem);
      }
      asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
      ascend_nd2nz_scatter<48, 64, bfloat16_t, bfloat16_t>((__ubuf__ bfloat16_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[52352]))), (__ubuf__ bfloat16_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[46080]))));
      asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(1));
      ascend_nd2nz_scatter<48, 64, bfloat16_t, bfloat16_t>((__ubuf__ bfloat16_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[55424]))), (__ubuf__ bfloat16_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[49216]))));
      asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
      asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(1));
      asc_copy_ub2l1((__cbuf__ void*)(((__cbuf__ bfloat16_t*)buf_dyn_l1 + ((sid * 768) + 102400))), (__ubuf__ void*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[46080]))), 4, 48, 1, 48);
      asc_sync_intra_arrive(PIPE_MTE3, 1);
      asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
      asc_copy_ub2l1((__cbuf__ void*)(((__cbuf__ bfloat16_t*)buf_dyn_l1 + ((sid * 768) + 108544))), (__ubuf__ void*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[49216]))), 4, 48, 1, 48);
      asc_sync_intra_arrive(PIPE_MTE3, 0);
      asc_set_atomic_add_float();
      asc_set_atomic_none();
    }
    asc_sync_intra_wait(PIPE_V, 3);
    asc_vf_call<_bwd_main_kernel_simt_vf_2>(cce::dim3(128), buf_dyn_shmem);
    asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(2));
    asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(2));
    if ((((((int32_t)block_idx) % 6) * 3) + ((((int32_t)sid_1) * 3) >> 1)) < 16) {
      asc_copy_ub2gm_align((__gm__ uint8_t*)((&(dk[((((((int64_t)((int32_t)block_idx)) % (int64_t)6) * (int64_t)36864) + (sid_1 * (int64_t)18432)) + ((((int64_t)((int32_t)block_idx)) / (int64_t)6) * (int64_t)192))]))), (__ubuf__ uint8_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[30720]))), min(48, max(((512 - (((int32_t)sid_1) * 48)) - ((((int32_t)block_idx) % 6) * 96)), 0)), 384, static_cast<asc_store_l2_cache_mode>(4), 768, 384);
      asc_copy_ub2gm_align((__gm__ uint8_t*)((&(dv[((((((int64_t)((int32_t)block_idx)) % (int64_t)6) * (int64_t)24576) + (sid_1 * (int64_t)12288)) + ((((int64_t)((int32_t)block_idx)) / (int64_t)6) * (int64_t)128))]))), (__ubuf__ uint8_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[39936]))), min(48, max(((512 - (((int32_t)sid_1) * 48)) - ((((int32_t)block_idx) % 6) * 96)), 0)), 256, static_cast<asc_store_l2_cache_mode>(4), 512, 256);
    }
    asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  }
}

