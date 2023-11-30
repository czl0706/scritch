// tvm target: c -keys=cpu -model=esp32
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_concatenate_cast(int8_t* p0, int8_t* p1, int8_t* p2, int16_t* T_cast, uint8_t* global_const_workspace_14_var, uint8_t* global_workspace_15_var) {
  void* concatenate_ext_let = (&(global_workspace_15_var[0]));
  for (int32_t j = 0; j < 60; ++j) {
    ((int8_t*)concatenate_ext_let)[j] = p0[j];
  }
  for (int32_t j_1 = 0; j_1 < 60; ++j_1) {
    ((int8_t*)concatenate_ext_let)[(j_1 + 60)] = p1[j_1];
  }
  for (int32_t j_2 = 0; j_2 < 60; ++j_2) {
    ((int8_t*)concatenate_ext_let)[(j_2 + 120)] = p2[j_2];
  }
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 180; ++ax0_ax1_fused) {
    T_cast[ax0_ax1_fused] = ((int16_t)((int8_t*)concatenate_ext_let)[ax0_ax1_fused]);
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_where_add_righ_2dc27ba01d5ce99b_(int16_t* p0, int8_t* T_relu, uint8_t* global_const_workspace_8_var, uint8_t* global_workspace_9_var) {
  void* fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_constant_3_let = (&(global_const_workspace_8_var[65760]));
  void* fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_constant_2_let = (&(global_const_workspace_8_var[66240]));
  void* fused_nn_contrib_dense_pack_constant_1_let = (&(global_const_workspace_8_var[67920]));
  void* fused_constant_1_let = (&(global_const_workspace_8_var[18000]));
  for (int32_t ax1_outer_ax0_outer_fused = 0; ax1_outer_ax0_outer_fused < 5; ++ax1_outer_ax0_outer_fused) {
    void* compute_let = (&(global_workspace_9_var[432]));
    for (int32_t y_inner_outer_x_inner_outer_fused = 0; y_inner_outer_x_inner_outer_fused < 2; ++y_inner_outer_x_inner_outer_fused) {
      void* compute_global_let = (&(global_workspace_9_var[480]));
      for (int32_t x_c_init = 0; x_c_init < 6; ++x_c_init) {
        ((int32_t*)compute_global_let)[x_c_init] = 0;
      }
      for (int32_t k_outer = 0; k_outer < 150; ++k_outer) {
        for (int32_t x_c = 0; x_c < 6; ++x_c) {
          ((int32_t*)compute_global_let)[x_c] = (((int32_t*)compute_global_let)[x_c] + (((int32_t)p0[k_outer]) * ((int32_t)((int16_t*)fused_constant_1_let)[((((ax1_outer_ax0_outer_fused * 1800) + (y_inner_outer_x_inner_outer_fused * 900)) + (k_outer * 6)) + x_c)])));
        }
      }
      for (int32_t x_inner_inner = 0; x_inner_inner < 6; ++x_inner_inner) {
        ((int32_t*)compute_let)[((y_inner_outer_x_inner_outer_fused * 6) + x_inner_inner)] = ((int32_t*)compute_global_let)[x_inner_inner];
      }
    }
    for (int32_t ax1_inner_outer = 0; ax1_inner_outer < 2; ++ax1_inner_outer) {
      for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 6; ++ax1_inner_inner) {
        int32_t cse_var_3 = (ax1_inner_outer * 6);
        int32_t cse_var_2 = (cse_var_3 + ax1_inner_inner);
        int32_t cse_var_1 = (((ax1_outer_ax0_outer_fused * 12) + cse_var_3) + ax1_inner_inner);
        int32_t v_ = (int32_t)((((((int64_t)((int32_t*)compute_let)[cse_var_2]) * (int64_t)1073741824) + (((int64_t)((int32_t*)fused_nn_contrib_dense_pack_constant_1_let)[cse_var_1]) * (int64_t)1073741824)) + ((0 < ((int32_t)((int64_t)0 <= (((int64_t)((int32_t*)compute_let)[cse_var_2]) + ((int64_t)((int32_t*)fused_nn_contrib_dense_pack_constant_1_let)[cse_var_1]))))) ? ((int64_t*)fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_constant_2_let)[cse_var_1] : ((int64_t*)fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_constant_3_let)[cse_var_1])) >> (int64_t)40);
        int32_t v__1 = (v_) < (127) ? (v_) : (127);
        int8_t v__2 = (int8_t)((v__1) > (-128) ? (v__1) : (-128));
        int8_t v__3 = (int8_t)0;
        T_relu[cse_var_1] = ((v__2) > (v__3) ? (v__2) : (v__3));
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_where_add_righ_44f31c9e7825803c_(int16_t* p0, int8_t* T_cast, uint8_t* global_const_workspace_4_var, uint8_t* global_workspace_5_var) {
  void* fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_constant_1_let = (&(global_const_workspace_4_var[66720]));
  void* fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_constant_let = (&(global_const_workspace_4_var[67200]));
  void* fused_nn_contrib_dense_pack_constant_let = (&(global_const_workspace_4_var[68160]));
  void* fused_constant_let = (&(global_const_workspace_4_var[36000]));
  for (int32_t ax1_outer_ax0_outer_fused = 0; ax1_outer_ax0_outer_fused < 5; ++ax1_outer_ax0_outer_fused) {
    void* compute_let = (&(global_workspace_5_var[368]));
    for (int32_t y_inner_outer_x_inner_outer_fused = 0; y_inner_outer_x_inner_outer_fused < 2; ++y_inner_outer_x_inner_outer_fused) {
      void* compute_global_let = (&(global_workspace_5_var[416]));
      for (int32_t x_c_init = 0; x_c_init < 6; ++x_c_init) {
        ((int32_t*)compute_global_let)[x_c_init] = 0;
      }
      for (int32_t k_outer = 0; k_outer < 150; ++k_outer) {
        for (int32_t x_c = 0; x_c < 6; ++x_c) {
          ((int32_t*)compute_global_let)[x_c] = (((int32_t*)compute_global_let)[x_c] + (((int32_t)p0[k_outer]) * ((int32_t)((int16_t*)fused_constant_let)[((((ax1_outer_ax0_outer_fused * 1800) + (y_inner_outer_x_inner_outer_fused * 900)) + (k_outer * 6)) + x_c)])));
        }
      }
      for (int32_t x_inner_inner = 0; x_inner_inner < 6; ++x_inner_inner) {
        ((int32_t*)compute_let)[((y_inner_outer_x_inner_outer_fused * 6) + x_inner_inner)] = ((int32_t*)compute_global_let)[x_inner_inner];
      }
    }
    for (int32_t ax1_inner_outer = 0; ax1_inner_outer < 2; ++ax1_inner_outer) {
      for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 6; ++ax1_inner_inner) {
        int32_t cse_var_3 = (ax1_inner_outer * 6);
        int32_t cse_var_2 = (cse_var_3 + ax1_inner_inner);
        int32_t cse_var_1 = (((ax1_outer_ax0_outer_fused * 12) + cse_var_3) + ax1_inner_inner);
        int32_t v_ = (int32_t)((((((int64_t)((int32_t*)compute_let)[cse_var_2]) * (int64_t)1073741824) + (((int64_t)((int32_t*)fused_nn_contrib_dense_pack_constant_let)[cse_var_1]) * (int64_t)1073741824)) + ((0 < ((int32_t)((int64_t)0 <= (((int64_t)((int32_t*)compute_let)[cse_var_2]) + ((int64_t)((int32_t*)fused_nn_contrib_dense_pack_constant_let)[cse_var_1]))))) ? ((int64_t*)fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_constant_let)[cse_var_1] : ((int64_t*)fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_constant_1_let)[cse_var_1])) >> (int64_t)39);
        int32_t v__1 = (v_) < (127) ? (v_) : (127);
        int8_t v__2 = (int8_t)((v__1) > (-128) ? (v__1) : (-128));
        int8_t v__3 = (int8_t)0;
        int32_t v__4 = (((int32_t)((v__2) > (v__3) ? (v__2) : (v__3))) + 1) >> 1;
        int32_t v__5 = (v__4) < (127) ? (v__4) : (127);
        T_cast[cse_var_1] = ((int8_t)((v__5) > (-128) ? (v__5) : (-128)));
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_where_add_righ_44f31c9e7825803c__1(int16_t* p0, int8_t* T_cast, uint8_t* global_const_workspace_12_var, uint8_t* global_workspace_13_var) {
  void* fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_constant_5_let = (&(global_const_workspace_12_var[64800]));
  void* fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_constant_4_let = (&(global_const_workspace_12_var[65280]));
  void* fused_nn_contrib_dense_pack_constant_2_let = (&(global_const_workspace_12_var[67680]));
  void* fused_constant_2_let = (&(global_const_workspace_12_var[0]));
  for (int32_t ax1_outer_ax0_outer_fused = 0; ax1_outer_ax0_outer_fused < 5; ++ax1_outer_ax0_outer_fused) {
    void* compute_let = (&(global_workspace_13_var[496]));
    for (int32_t y_inner_outer_x_inner_outer_fused = 0; y_inner_outer_x_inner_outer_fused < 2; ++y_inner_outer_x_inner_outer_fused) {
      void* compute_global_let = (&(global_workspace_13_var[544]));
      for (int32_t x_c_init = 0; x_c_init < 6; ++x_c_init) {
        ((int32_t*)compute_global_let)[x_c_init] = 0;
      }
      for (int32_t k_outer = 0; k_outer < 150; ++k_outer) {
        for (int32_t x_c = 0; x_c < 6; ++x_c) {
          ((int32_t*)compute_global_let)[x_c] = (((int32_t*)compute_global_let)[x_c] + (((int32_t)p0[k_outer]) * ((int32_t)((int16_t*)fused_constant_2_let)[((((ax1_outer_ax0_outer_fused * 1800) + (y_inner_outer_x_inner_outer_fused * 900)) + (k_outer * 6)) + x_c)])));
        }
      }
      for (int32_t x_inner_inner = 0; x_inner_inner < 6; ++x_inner_inner) {
        ((int32_t*)compute_let)[((y_inner_outer_x_inner_outer_fused * 6) + x_inner_inner)] = ((int32_t*)compute_global_let)[x_inner_inner];
      }
    }
    for (int32_t ax1_inner_outer = 0; ax1_inner_outer < 2; ++ax1_inner_outer) {
      for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 6; ++ax1_inner_inner) {
        int32_t cse_var_3 = (ax1_inner_outer * 6);
        int32_t cse_var_2 = (cse_var_3 + ax1_inner_inner);
        int32_t cse_var_1 = (((ax1_outer_ax0_outer_fused * 12) + cse_var_3) + ax1_inner_inner);
        int32_t v_ = (int32_t)((((((int64_t)((int32_t*)compute_let)[cse_var_2]) * (int64_t)1073741824) + (((int64_t)((int32_t*)fused_nn_contrib_dense_pack_constant_2_let)[cse_var_1]) * (int64_t)1073741824)) + ((0 < ((int32_t)((int64_t)0 <= (((int64_t)((int32_t*)compute_let)[cse_var_2]) + ((int64_t)((int32_t*)fused_nn_contrib_dense_pack_constant_2_let)[cse_var_1]))))) ? ((int64_t*)fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_constant_4_let)[cse_var_1] : ((int64_t*)fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_constant_5_let)[cse_var_1])) >> (int64_t)39);
        int32_t v__1 = (v_) < (127) ? (v_) : (127);
        int8_t v__2 = (int8_t)((v__1) > (-128) ? (v__1) : (-128));
        int8_t v__3 = (int8_t)0;
        int32_t v__4 = (((int32_t)((v__2) > (v__3) ? (v__2) : (v__3))) + 1) >> 1;
        int32_t v__5 = (v__4) < (127) ? (v__4) : (127);
        T_cast[cse_var_1] = ((int8_t)((v__5) > (-128) ? (v__5) : (-128)));
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_where_add_righ_467e114bd6595de9_(int16_t* p0, float* T_multiply, uint8_t* global_const_workspace_18_var, uint8_t* global_workspace_19_var) {
  void* fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_constant_9_let = (&(global_const_workspace_18_var[69136]));
  void* fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_constant_8_let = (&(global_const_workspace_18_var[69152]));
  void* fused_nn_contrib_dense_pack_constant_4_let = (&(global_const_workspace_18_var[69168]));
  void* fused_constant_4_let = (&(global_const_workspace_18_var[69008]));
  void* compute_let = (&(global_workspace_19_var[0]));
  void* compute_global_let = (&(global_workspace_19_var[64]));
  for (int32_t x_c_init = 0; x_c_init < 2; ++x_c_init) {
    ((int32_t*)compute_global_let)[x_c_init] = 0;
  }
  for (int32_t k_outer = 0; k_outer < 30; ++k_outer) {
    for (int32_t x_c = 0; x_c < 2; ++x_c) {
      ((int32_t*)compute_global_let)[x_c] = (((int32_t*)compute_global_let)[x_c] + (((int32_t)p0[k_outer]) * ((int32_t)((int16_t*)fused_constant_4_let)[((k_outer * 2) + x_c)])));
    }
  }
  for (int32_t x_inner_inner = 0; x_inner_inner < 2; ++x_inner_inner) {
    ((int32_t*)compute_let)[x_inner_inner] = ((int32_t*)compute_global_let)[x_inner_inner];
  }
  for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 2; ++ax1_inner_inner) {
    int32_t v_ = (int32_t)((((((int64_t)((int32_t*)compute_let)[ax1_inner_inner]) * (int64_t)1073741824) + (((int64_t)((int32_t*)fused_nn_contrib_dense_pack_constant_4_let)[ax1_inner_inner]) * (int64_t)1073741824)) + ((0 < ((int32_t)((int64_t)0 <= (((int64_t)((int32_t*)compute_let)[ax1_inner_inner]) + ((int64_t)((int32_t*)fused_nn_contrib_dense_pack_constant_4_let)[ax1_inner_inner]))))) ? ((int64_t*)fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_constant_8_let)[ax1_inner_inner] : ((int64_t*)fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_constant_9_let)[ax1_inner_inner])) >> (int64_t)37);
    int32_t v__1 = (v_) < (127) ? (v_) : (127);
    T_multiply[ax1_inner_inner] = (((float)((int8_t)((v__1) > (-128) ? (v__1) : (-128)))) * 1.953125e-03f);
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_where_add_righ_c187d42c02fffb00_(int16_t* p0, int16_t* T_cast, uint8_t* global_const_workspace_16_var, uint8_t* global_workspace_17_var) {
  void* fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_constant_7_let = (&(global_const_workspace_16_var[68400]));
  void* fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_constant_6_let = (&(global_const_workspace_16_var[68640]));
  void* fused_nn_contrib_dense_pack_constant_3_let = (&(global_const_workspace_16_var[68880]));
  void* fused_constant_3_let = (&(global_const_workspace_16_var[54000]));
  for (int32_t ax1_outer_ax0_outer_fused = 0; ax1_outer_ax0_outer_fused < 5; ++ax1_outer_ax0_outer_fused) {
    void* compute_let = (&(global_workspace_17_var[64]));
    void* compute_global_let = (&(global_workspace_17_var[96]));
    for (int32_t x_c_init = 0; x_c_init < 6; ++x_c_init) {
      ((int32_t*)compute_global_let)[x_c_init] = 0;
    }
    for (int32_t k_outer = 0; k_outer < 180; ++k_outer) {
      for (int32_t x_c = 0; x_c < 6; ++x_c) {
        ((int32_t*)compute_global_let)[x_c] = (((int32_t*)compute_global_let)[x_c] + (((int32_t)p0[k_outer]) * ((int32_t)((int16_t*)fused_constant_3_let)[(((ax1_outer_ax0_outer_fused * 1080) + (k_outer * 6)) + x_c)])));
      }
    }
    for (int32_t x_inner_inner = 0; x_inner_inner < 6; ++x_inner_inner) {
      ((int32_t*)compute_let)[x_inner_inner] = ((int32_t*)compute_global_let)[x_inner_inner];
    }
    for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 6; ++ax1_inner_inner) {
      int32_t cse_var_1 = ((ax1_outer_ax0_outer_fused * 6) + ax1_inner_inner);
      int32_t v_ = (int32_t)((((((int64_t)((int32_t*)compute_let)[ax1_inner_inner]) * (int64_t)1073741824) + (((int64_t)((int32_t*)fused_nn_contrib_dense_pack_constant_3_let)[cse_var_1]) * (int64_t)1073741824)) + ((0 < ((int32_t)((int64_t)0 <= (((int64_t)((int32_t*)compute_let)[ax1_inner_inner]) + ((int64_t)((int32_t*)fused_nn_contrib_dense_pack_constant_3_let)[cse_var_1]))))) ? ((int64_t*)fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_constant_6_let)[cse_var_1] : ((int64_t*)fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_constant_7_let)[cse_var_1])) >> (int64_t)38);
      int32_t v__1 = (v_) < (127) ? (v_) : (127);
      int8_t v__2 = (int8_t)((v__1) > (-128) ? (v__1) : (-128));
      int8_t v__3 = (int8_t)0;
      T_cast[cse_var_1] = ((int16_t)((v__2) > (v__3) ? (v__2) : (v__3)));
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_strided_slice_divide_round_clip_cast_cast(float* p0, int16_t* T_cast, uint8_t* global_const_workspace_2_var, uint8_t* global_workspace_3_var) {
  for (int32_t ax1_outer = 0; ax1_outer < 10; ++ax1_outer) {
    for (int32_t ax1_inner = 0; ax1_inner < 16; ++ax1_inner) {
      if (((ax1_outer * 8) + (ax1_inner >> 1)) < 75) {
        int32_t cse_var_1 = ((ax1_outer * 16) + ax1_inner);
        float v_ = roundf((p0[cse_var_1] * 3.200000e+01f));
        float v__1 = (v_) < (1.270000e+02f) ? (v_) : (1.270000e+02f);
        T_cast[cse_var_1] = ((int16_t)((int8_t)((v__1) > (-1.280000e+02f) ? (v__1) : (-1.280000e+02f))));
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_strided_slice_divide_round_clip_cast_cast_1(float* p0, int16_t* T_cast, uint8_t* global_const_workspace_6_var, uint8_t* global_workspace_7_var) {
  for (int32_t ax1_outer = 0; ax1_outer < 10; ++ax1_outer) {
    for (int32_t ax1_inner = 0; ax1_inner < 16; ++ax1_inner) {
      if (((ax1_outer * 8) + (ax1_inner >> 1)) < 75) {
        int32_t cse_var_1 = ((ax1_outer * 16) + ax1_inner);
        float v_ = roundf((p0[(cse_var_1 + 150)] * 3.200000e+01f));
        float v__1 = (v_) < (1.270000e+02f) ? (v_) : (1.270000e+02f);
        T_cast[cse_var_1] = ((int16_t)((int8_t)((v__1) > (-1.280000e+02f) ? (v__1) : (-1.280000e+02f))));
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_strided_slice_divide_round_clip_cast_cast_2(float* p0, int16_t* T_cast, uint8_t* global_const_workspace_10_var, uint8_t* global_workspace_11_var) {
  for (int32_t ax1_outer = 0; ax1_outer < 10; ++ax1_outer) {
    for (int32_t ax1_inner = 0; ax1_inner < 16; ++ax1_inner) {
      if (((ax1_outer * 8) + (ax1_inner >> 1)) < 75) {
        int32_t cse_var_1 = ((ax1_outer * 16) + ax1_inner);
        float v_ = roundf((p0[(cse_var_1 + 300)] * 3.200000e+01f));
        float v__1 = (v_) < (1.270000e+02f) ? (v_) : (1.270000e+02f);
        T_cast[cse_var_1] = ((int16_t)((int8_t)((v__1) > (-1.280000e+02f) ? (v__1) : (-1.280000e+02f))));
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(float* input_buffer_var, float* output_buffer_var, uint8_t* global_const_workspace_0_var, uint8_t* global_workspace_1_var) {
  void* sid_6_let = (&(global_workspace_1_var[432]));
  void* sid_4_let = (&(global_workspace_1_var[368]));
  void* sid_2_let = (&(global_workspace_1_var[304]));
  void* sid_1_let = (&(global_workspace_1_var[0]));
  void* sid_3_let = (&(global_workspace_1_var[0]));
  void* sid_5_let = (&(global_workspace_1_var[0]));
  void* sid_7_let = (&(global_workspace_1_var[192]));
  void* sid_8_let = (&(global_workspace_1_var[0]));
  if (tvmgen_default_fused_strided_slice_divide_round_clip_cast_cast(input_buffer_var, sid_1_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_where_add_righ_44f31c9e7825803c_(sid_1_let, sid_2_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_strided_slice_divide_round_clip_cast_cast_1(input_buffer_var, sid_3_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_where_add_righ_2dc27ba01d5ce99b_(sid_3_let, sid_4_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_strided_slice_divide_round_clip_cast_cast_2(input_buffer_var, sid_5_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_where_add_righ_44f31c9e7825803c__1(sid_5_let, sid_6_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_concatenate_cast(sid_2_let, sid_4_let, sid_6_let, sid_7_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_where_add_righ_c187d42c02fffb00_(sid_7_let, sid_8_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_dense_pack_add_cast_multiply_zeros_greater_equal_where_add_righ_467e114bd6595de9_(sid_8_let, output_buffer_var, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  return 0;
}

