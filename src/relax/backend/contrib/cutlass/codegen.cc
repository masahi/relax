/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relax/backend/contrib/cutlass/codegen.cc
 * \brief Implementation of the CUTLASS JSON serializer.
 */
#include <tvm/ir/module.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/nn.h>
#include <tvm/relax/type.h>

#include <memory>
#include <string>
#include <vector>

#include "../codegen_json/codegen_json.h"
#include "../utils.h"

namespace tvm {
namespace relax {
namespace contrib {

using Str2StrMap = std::unordered_map<std::string, std::string>;

static Str2StrMap dtype_map = {{"float16", "cutlass::half_t"},
                               {"float32", "float"},
                               {"int8", "int8_t"},
                               {"uint8", "uint8_t"},
                               {"int32", "int32_t"}};

constexpr const char* kAnyDim = "Any";

std::string GetDimAsStr(ObjectRef dim) {
  if (auto d = dim.as<IntImmNode>()) {
    return std::to_string(d->value);
  }
  return kAnyDim;
}

inline void CutlassPrint(std::ostringstream& os, const std::string& stmt, int indent = 2) {
  for (int i = 0; i < indent; ++i) {
    os << " ";
  }
  os << stmt;
}

Str2StrMap ArgsCommon(const Map<String, ObjectRef>& attrs) {
  Str2StrMap args;
  auto arg0_dtype = std::string(attrs["arg0_dtype"].as<StringObj>()->data);
  auto arg1_dtype = std::string(attrs["arg1_dtype"].as<StringObj>()->data);
  auto ret_dtype = std::string(attrs["ret_dtype"].as<StringObj>()->data);
  args["ElementInputA"] = dtype_map.at(arg0_dtype);
  args["ElementInputB"] = dtype_map.at(arg1_dtype);
  args["ElementOutput"] = dtype_map.at(ret_dtype);
  args["op_def"] = std::string(attrs["cutlass_op_def"].as<StringObj>()->data);
  args["op_name"] = std::string(attrs["cutlass_op_name"].as<StringObj>()->data);
  args["op_type"] = std::string(attrs["op_type"].as<StringObj>()->data);
  return args;
}

Str2StrMap GemmArgsCommon(const Map<String, ObjectRef>& attrs) {
  Str2StrMap args = ArgsCommon(attrs);
  args["lda"] = std::string(attrs["lda"].as<StringObj>()->data);
  args["ldb"] = std::string(attrs["ldb"].as<StringObj>()->data);
  args["ldc"] = std::string(attrs["ldc"].as<StringObj>()->data);
  return args;
}

Str2StrMap DenseArgs(const Map<String, ObjectRef>& attrs) {
  Str2StrMap args = GemmArgsCommon(attrs);
  auto arg0_shape = attrs["arg0_shape"].as<ArrayNode>();
  auto arg1_shape = attrs["arg1_shape"].as<ArrayNode>();
  args["M"] = GetDimAsStr(arg0_shape->at(0));
  args["K"] = GetDimAsStr(arg0_shape->at(1));
  args["N"] = GetDimAsStr(arg1_shape->at(0));
  return args;
}

Str2StrMap BatchMatmulArgs(const Map<String, ObjectRef>& attrs) {
  Str2StrMap args = GemmArgsCommon(attrs);
  args["batch"] = GetDimAsStr(attrs["batch"]);
  args["batch_stride_A"] = GetDimAsStr(attrs["batch_stride_A"]);
  args["batch_stride_B"] = GetDimAsStr(attrs["batch_stride_B"]);
  args["batch_stride_C"] = GetDimAsStr(attrs["batch_stride_C"]);
  auto arg0_shape = attrs["arg0_shape"].as<ArrayNode>();
  auto arg1_shape = attrs["arg1_shape"].as<ArrayNode>();
  args["M"] = GetDimAsStr(arg0_shape->at(1));
  args["K"] = GetDimAsStr(arg0_shape->at(2));
  args["N"] = GetDimAsStr(arg1_shape->at(1));
  return args;
}

void AppendPrologue(std::ostringstream& gemm_decl, const Str2StrMap& attrs,
                    const std::vector<std::string>& func_args, const std::string& kernel,
                    bool has_bias, bool is_gelu, int m_axis_idx, int n_axis_idx, int k_axis_idx) {
  CutlassPrint(gemm_decl, "using ElementInputA = " + attrs.at("ElementInputA") + ";\n");
  CutlassPrint(gemm_decl, "using ElementInputB = " + attrs.at("ElementInputB") + ";\n");
  CutlassPrint(gemm_decl, "using ElementOutput = " + attrs.at("ElementOutput") + ";\n");
  CutlassPrint(gemm_decl, "using ElementComputeEpilogue = " + attrs.at("ElementOutput") + ";\n");
  CutlassPrint(gemm_decl, attrs.at("op_def"));
  CutlassPrint(gemm_decl, "using " + kernel + " = Operation_" + attrs.at("op_name") + ";\n");

  auto get_dim = [&attrs, &func_args](const std::string& axis, int arg_idx, int axis_idx) {
    if (attrs.at(axis) == kAnyDim) {
      return func_args[arg_idx] + "->shape[" + std::to_string(axis_idx) + "]";
    } else {
      return attrs.at(axis);
    }
  };
  CutlassPrint(gemm_decl, "int M = " + get_dim("M", 0, m_axis_idx) + ";\n");
  CutlassPrint(gemm_decl, "int N = " + get_dim("N", 1, n_axis_idx) + ";\n");
  CutlassPrint(gemm_decl, "int K = " + get_dim("K", 0, k_axis_idx) + ";\n");
  CutlassPrint(gemm_decl, "cutlass::gemm::GemmCoord problem_size(M, N, K);\n");
  CutlassPrint(gemm_decl, "ElementComputeEpilogue alpha = ElementComputeEpilogue(1);\n");
  if (is_gelu) {
    // GeLU epilogue does not compile with NoBetaScaling, so we explicitly specify the scale.
    CutlassPrint(gemm_decl, "ElementComputeEpilogue beta = ElementComputeEpilogue(1);\n");
  } else {
    CutlassPrint(gemm_decl, "ElementComputeEpilogue beta = ElementComputeEpilogue(0);\n");
  }

  ICHECK(func_args.size() >= 2);
  CutlassPrint(gemm_decl, "void* ptr_a = (void*)(" + func_args[0] + "->data);\n");
  CutlassPrint(gemm_decl, "void* ptr_b = (void*)(" + func_args[1] + "->data);\n");
  if (has_bias) {
    ICHECK(func_args.size() >= 3);
    CutlassPrint(gemm_decl, "void* ptr_c_bias = (void*)(" + func_args[2] + "->data);\n");
  }

  CutlassPrint(gemm_decl, "void* ptr_out = (void*)(out0->data);\n");

  CutlassPrint(gemm_decl, "typename " + kernel + "::Arguments arguments{\n");
  CutlassPrint(gemm_decl, " problem_size,\n");
}

void AppendGemmExecute(std::ostringstream& gemm_decl, const std::string& kernel) {
  // Using the arguments, query for extra workspace required for matrix multiplication computation
  CutlassPrint(gemm_decl,
               "size_t workspace_size = " + kernel + "::get_workspace_size(arguments);\n");
  // Allocate workspace memory
  CutlassPrint(gemm_decl,
               "cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);\n");
  // Instantiate CUTLASS kernel depending on template
  CutlassPrint(gemm_decl, kernel + " gemm_op;\n");

  // Check the problem size is supported or not
  CutlassPrint(gemm_decl, "cutlass::Status status = gemm_op.can_implement(arguments);\n");
  CutlassPrint(gemm_decl, "CHECK(status == cutlass::Status::kSuccess);\n");
  // Initialize CUTLASS kernel with arguments and workspace pointer
  CutlassPrint(gemm_decl, "status = gemm_op.initialize(arguments, workspace.get());\n");
  CutlassPrint(gemm_decl, "CHECK(status == cutlass::Status::kSuccess);\n");
  // Launch initialized CUTLASS kernel
  CutlassPrint(gemm_decl, "status = gemm_op();\n");
  CutlassPrint(gemm_decl, "CHECK(status == cutlass::Status::kSuccess);\n");
}

std::string DenseOp(std::string id, const Str2StrMap& attrs,
                    const std::vector<std::string>& func_args) {
  bool has_bias = attrs.at("op_type").find("bias") != std::string::npos;
  bool is_gelu =
      attrs.at("op_type").find("cutlass.dense_bias_gelu") != std::string::npos;  // fp32 or fp16
  std::ostringstream gemm_decl;
  AppendPrologue(gemm_decl, attrs, func_args, "Gemm", has_bias, is_gelu, 0, 0, 1);

  CutlassPrint(gemm_decl, " {static_cast<ElementInputA*>(ptr_a), " + attrs.at("lda") + "},\n");
  CutlassPrint(gemm_decl, " {static_cast<ElementInputB*>(ptr_b), " + attrs.at("ldb") + "},\n");
  if (has_bias) {
    CutlassPrint(gemm_decl, " {static_cast<ElementOutput*>(ptr_c_bias), 0},\n");
  } else {
    CutlassPrint(gemm_decl, " {static_cast<ElementOutput*>(ptr_out), " + attrs.at("ldc") + "},\n");
  }
  CutlassPrint(gemm_decl, " {static_cast<ElementOutput*>(ptr_out), " + attrs.at("ldc") + "},\n");
  if (has_bias && !is_gelu) {
    CutlassPrint(gemm_decl, " {alpha},\n");
  } else {
    // For GeLU, we explicitly specify the scale.
    CutlassPrint(gemm_decl, " {alpha, beta},\n");
  }
  CutlassPrint(gemm_decl, " 1};\n");  // split_k_slices

  AppendGemmExecute(gemm_decl, "Gemm");
  return gemm_decl.str();
}

std::string BatchMatmulOp(std::string id, const Str2StrMap& attrs,
                          const std::vector<std::string>& func_args) {
  std::ostringstream gemm_decl;
  AppendPrologue(gemm_decl, attrs, func_args, "BatchedGemm", false, false, 1, 1, 2);

  auto get_batch_stride = [&attrs, &func_args](const std::string& name, int arg0_idx, int arg1_idx,
                                               int arg0_axis_idx, int arg1_axis_idx) {
    if (attrs.at(name) == kAnyDim) {
      return func_args[arg0_idx] + "->shape[" + std::to_string(arg0_axis_idx) + "] * " +
             func_args[arg1_idx] + "->shape[" + std::to_string(arg1_axis_idx) + "]";
    } else {
      return attrs.at(name);
    }
  };

  CutlassPrint(gemm_decl, " {static_cast<ElementInputA*>(ptr_a), " + attrs.at("lda") + "},\n");
  CutlassPrint(gemm_decl, get_batch_stride("batch_stride_A", 0, 0, 1, 2) + ",\n");
  CutlassPrint(gemm_decl, " {static_cast<ElementInputB*>(ptr_b), " + attrs.at("ldb") + "},\n");
  CutlassPrint(gemm_decl, get_batch_stride("batch_stride_B", 1, 1, 1, 2) + ",\n");
  CutlassPrint(gemm_decl, " {static_cast<ElementOutput*>(ptr_out), " + attrs.at("ldc") + "},\n");
  CutlassPrint(gemm_decl, get_batch_stride("batch_stride_C", 0, 1, 1, 1) + ",\n");
  CutlassPrint(gemm_decl, " {static_cast<ElementOutput*>(ptr_out), " + attrs.at("ldc") + "},\n");
  CutlassPrint(gemm_decl, get_batch_stride("batch_stride_C", 0, 1, 1, 1) + ",\n");
  CutlassPrint(gemm_decl, " {alpha, beta},\n");

  if (attrs.at("batch") == kAnyDim) {
    CutlassPrint(gemm_decl, func_args[0] + "->shape[0]" + "};\n");
  } else {
    CutlassPrint(gemm_decl, attrs.at("batch") + "};\n");
  }

  AppendGemmExecute(gemm_decl, "BatchedGemm");
  return gemm_decl.str();
}

Str2StrMap Conv2dArgs(const Map<String, ObjectRef>& attrs, bool is_dgrad = false,
                      bool is_wgrad = false) {
  Str2StrMap args = ArgsCommon(attrs);
  auto arg0_shape = attrs["arg0_shape"].as<ArrayNode>();
  auto arg1_shape = attrs["arg1_shape"].as<ArrayNode>();
  auto ret_shape = attrs["ret_shape"].as<ArrayNode>();
  auto activation_shape = arg0_shape;
  auto weight_shape = arg1_shape;
  auto output_shape = ret_shape;

  if (is_dgrad) {
    activation_shape = ret_shape;
    output_shape = arg0_shape;
  } else if (is_wgrad) {
    activation_shape = arg1_shape;
    weight_shape = ret_shape;
    output_shape = arg0_shape;
  }

  args["N"] = GetDimAsStr(activation_shape->at(0));
  args["H"] = GetDimAsStr(activation_shape->at(1));
  args["W"] = GetDimAsStr(activation_shape->at(2));
  args["C"] = GetDimAsStr(activation_shape->at(3));
  args["P"] = GetDimAsStr(output_shape->at(1));
  args["Q"] = GetDimAsStr(output_shape->at(2));
  args["K"] = GetDimAsStr(output_shape->at(3));
  args["R"] = GetDimAsStr(weight_shape->at(1));
  args["S"] = GetDimAsStr(weight_shape->at(2));
  args["pad_h"] = GetDimAsStr(attrs["padding"].as<ArrayNode>()->at(0));
  args["pad_w"] = GetDimAsStr(attrs["padding"].as<ArrayNode>()->at(1));
  args["stride_h"] = GetDimAsStr(attrs["strides"].as<ArrayNode>()->at(0));
  args["stride_w"] = GetDimAsStr(attrs["strides"].as<ArrayNode>()->at(1));
  args["dilation_h"] = GetDimAsStr(attrs["dilation"].as<ArrayNode>()->at(0));
  args["dilation_w"] = GetDimAsStr(attrs["dilation"].as<ArrayNode>()->at(1));

  return args;
}

std::string Conv2dOp(std::string id, const Str2StrMap& attrs,
                     const std::vector<std::string>& func_args, bool has_residual_block = false) {
  auto op_type = attrs.at("op_type");
  bool has_bias = op_type.find("bias") != std::string::npos;
  bool no_bias_scaling = op_type != "cutlass.conv2d_bias_sigmoid" &&
                         op_type != "cutlass.conv2d_bias_silu" &&
                         op_type != "cutlass.conv2d_bias_hardswish";

  const std::string op_name = attrs.at("op_name");
  std::ostringstream conv2d_decl;
  CutlassPrint(conv2d_decl, attrs.at("op_def"));
  CutlassPrint(conv2d_decl, "using Operation_" + op_name +
                                " = cutlass::conv::device::ImplicitGemmConvolution<" + op_name +
                                ">;\n");
  CutlassPrint(conv2d_decl, "using Conv2d = Operation_" + op_name + ";\n");
  CutlassPrint(conv2d_decl, "using ElementInputA = Conv2d::ElementA;\n");
  CutlassPrint(conv2d_decl, "using ElementInputB = Conv2d::ElementB;\n");
  CutlassPrint(conv2d_decl, "using ElementComputeEpilogue = Conv2d::ElementAccumulator;\n");

  auto get_dim = [&attrs](const std::string& axis, const std::string& var_name, int axis_idx) {
    if (attrs.at(axis) == kAnyDim) {
      return var_name + "->shape[" + std::to_string(axis_idx) + "]";
    } else {
      return attrs.at(axis);
    }
  };

  CutlassPrint(conv2d_decl, "int N = " + get_dim("N", func_args[0], 0) + ";\n");
  CutlassPrint(conv2d_decl, "int H = " + get_dim("H", func_args[0], 1) + ";\n");
  CutlassPrint(conv2d_decl, "int W = " + get_dim("W", func_args[0], 2) + ";\n");
  CutlassPrint(conv2d_decl, "int C = " + attrs.at("C") + ";\n");
  CutlassPrint(conv2d_decl, "int K = " + attrs.at("K") + ";\n");
  CutlassPrint(conv2d_decl, "int R = " + attrs.at("R") + ";\n");
  CutlassPrint(conv2d_decl, "int S = " + attrs.at("S") + ";\n");
  CutlassPrint(conv2d_decl, "int P = " + get_dim("P", "out0", 1) + ";\n");
  CutlassPrint(conv2d_decl, "int Q = " + get_dim("Q", "out0", 2) + ";\n");
  CutlassPrint(conv2d_decl, "int pad_h = " + attrs.at("pad_h") + ";\n");
  CutlassPrint(conv2d_decl, "int pad_w = " + attrs.at("pad_w") + ";\n");
  CutlassPrint(conv2d_decl, "int stride_h = " + attrs.at("stride_h") + ";\n");
  CutlassPrint(conv2d_decl, "int stride_w = " + attrs.at("stride_w") + ";\n");
  CutlassPrint(conv2d_decl, "int dilation_h = " + attrs.at("dilation_h") + ";\n");
  CutlassPrint(conv2d_decl, "int dilation_w = " + attrs.at("dilation_w") + ";\n");

  const bool use_split_k = op_name.find("splitk") != std::string::npos;

  if (use_split_k) {
    std::string split_k_slices = op_name.substr(op_name.find_last_not_of("0123456789") + 1);
    CutlassPrint(conv2d_decl, "int split_k_slices = " + split_k_slices + ";\n");
  } else {
    CutlassPrint(conv2d_decl, "int split_k_slices = 1;\n");
  }

  CutlassPrint(
      conv2d_decl,
      "cutlass::conv::Conv2dProblemSize problem_size(N, H, W, C, K, R, S, P, Q, pad_h, pad_w, "
      "stride_h, stride_w, dilation_h, dilation_w, cutlass::conv::Mode::kCrossCorrelation, "
      "split_k_slices);\n");

  const std::string split_k_mode = use_split_k ? "kParallel" : "kSerial";
  CutlassPrint(conv2d_decl,
               "const cutlass::conv::SplitKMode split_k_mode = cutlass::conv::SplitKMode::" +
                   split_k_mode + ";\n");

  bool is_wgrad = op_type.find("backward_weight") != std::string::npos;
  bool is_dgrad = op_type.find("conv2d_transpose") != std::string::npos;

  ICHECK(func_args.size() >= 2);
  CutlassPrint(conv2d_decl, "void* ptr_a = (void*)(" + func_args[0] + "->data);\n");
  CutlassPrint(conv2d_decl, "void* ptr_b = (void*)(" + func_args[1] + "->data);\n");

  if (has_residual_block) {
    ICHECK(func_args.size() >= 4);
    CutlassPrint(conv2d_decl, "void* ptr_bias = (void*)(" + func_args[2] + "->data);\n");
    CutlassPrint(conv2d_decl, "void* ptr_residual = (void*)(" + func_args[3] + "->data);\n");
  } else if (has_bias) {
    ICHECK(func_args.size() >= 3);
    CutlassPrint(conv2d_decl, "void* ptr_c_bias = (void*)(" + func_args[2] + "->data);\n");
  }

  CutlassPrint(conv2d_decl, "void* ptr_out = (void*)(out0->data);\n");
  CutlassPrint(conv2d_decl, "ElementComputeEpilogue alpha = ElementComputeEpilogue(1);\n");
  if ((!has_bias || no_bias_scaling) && !has_residual_block) {
    CutlassPrint(conv2d_decl, "ElementComputeEpilogue beta = ElementComputeEpilogue(0);\n");
  } else {
    CutlassPrint(conv2d_decl, "ElementComputeEpilogue beta = ElementComputeEpilogue(1);\n");
  }
  CutlassPrint(conv2d_decl, "using cutlass::layout::TensorNHWC;\n");
  CutlassPrint(conv2d_decl,
               "auto activation_shape = TensorNHWC::packed(cutlass::make_Coord(N, H, W, C));\n");
  CutlassPrint(conv2d_decl,
               "auto weight_shape = TensorNHWC::packed(cutlass::make_Coord(K, R, S, C));\n");
  CutlassPrint(conv2d_decl,
               "auto output_oshape = TensorNHWC::packed(cutlass::make_Coord(N, P, Q, K));\n");

  if (is_wgrad) {
    CutlassPrint(conv2d_decl, "TensorNHWC layout_A(output_oshape);\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_B(activation_shape);\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_C(weight_shape);\n\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_D(weight_shape);\n\n");
  } else if (is_dgrad) {
    CutlassPrint(conv2d_decl, "TensorNHWC layout_A(output_oshape);\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_B(weight_shape);\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_C(activation_shape);\n\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_D(activation_shape);\n\n");
  } else {
    CutlassPrint(conv2d_decl, "TensorNHWC layout_A(activation_shape);\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_B(weight_shape);\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_C(output_oshape);\n\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_D(output_oshape);\n\n");
  }

  if (use_split_k) {
    CutlassPrint(conv2d_decl, "using ElementOutput = EpilogueOutputOp::ElementOutput;\n");
  } else {
    CutlassPrint(conv2d_decl, "using ElementOutput = Conv2d::ElementC;\n");
  }

  std::string tensor_c_init = "{static_cast<ElementOutput*>(ptr_out), layout_C}";
  if (has_residual_block) {
    tensor_c_init = "{static_cast<ElementOutput*>(ptr_residual), layout_C}";
  } else if (has_bias) {
    tensor_c_init =
        "{static_cast<ElementOutput*>(ptr_c_bias), cutlass::layout::TensorNHWC::Stride(0)}";
  }

  CutlassPrint(conv2d_decl,
               "cutlass::TensorRef<ElementOutput, TensorNHWC> tensor_c" + tensor_c_init + ";\n");
  CutlassPrint(conv2d_decl,
               "cutlass::TensorRef<ElementOutput, TensorNHWC> "
               "tensor_d{static_cast<ElementOutput*>(ptr_out),layout_D};\n");

  CutlassPrint(conv2d_decl, "typename Conv2d::Arguments arguments{\n");
  CutlassPrint(conv2d_decl, " problem_size,\n");
  CutlassPrint(conv2d_decl, " {static_cast<ElementInputA*>(ptr_a), layout_A},\n");
  CutlassPrint(conv2d_decl, " {static_cast<ElementInputB*>(ptr_b), layout_B},\n");

  if (use_split_k) {
    CutlassPrint(conv2d_decl, "{nullptr, TensorNHWC()},\n");
    CutlassPrint(conv2d_decl, "{nullptr, TensorNHWC()},\n");
  } else {
    CutlassPrint(conv2d_decl, " tensor_c,\n");
    CutlassPrint(conv2d_decl, " tensor_d,\n");
  }

  if (has_residual_block) {
    ICHECK(use_split_k == false) << "Split-k not supported for residual block fusion";
    CutlassPrint(conv2d_decl, "{alpha, beta},\n");
    CutlassPrint(conv2d_decl, "cutlass::conv::SplitKMode::kSerial,\n");  // split_k_slices
    CutlassPrint(conv2d_decl, "static_cast<ElementOutput*>(ptr_bias),\n");
    CutlassPrint(conv2d_decl, "nullptr, 0, K};\n");
  } else if (has_bias && no_bias_scaling) {
    CutlassPrint(conv2d_decl, " {alpha},\n");
    CutlassPrint(conv2d_decl, "split_k_mode\n};\n");
  } else {
    CutlassPrint(conv2d_decl, "{alpha, beta},\n");
    CutlassPrint(conv2d_decl, "split_k_mode\n};\n");
  }

  CutlassPrint(conv2d_decl, "Conv2d conv2d_op;\n");

  CutlassPrint(conv2d_decl, "size_t workspace_size = conv2d_op.get_workspace_size(arguments);\n");
  // Allocate workspace memory
  CutlassPrint(conv2d_decl,
               "cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);\n");
  // Check the problem size is supported or not
  CutlassPrint(conv2d_decl, "cutlass::Status status = conv2d_op.can_implement(arguments);\n");
  CutlassPrint(conv2d_decl, "CHECK(status == cutlass::Status::kSuccess);\n\n");

  if (use_split_k) {
    CutlassPrint(conv2d_decl,
                 "arguments.ref_D.reset(reinterpret_cast<ElementComputeEpilogue*>(workspace.get()),"
                 " layout_D);\n\n");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CutlassPrint(conv2d_decl, "status = conv2d_op.initialize(arguments, workspace.get());\n");
  CutlassPrint(conv2d_decl, "CHECK(status == cutlass::Status::kSuccess);\n\n");

  if (use_split_k) {
    CutlassPrint(
        conv2d_decl,
        "arguments.output_op = {ElementComputeEpilogue(1), ElementComputeEpilogue(0)}; \n");
    CutlassPrint(conv2d_decl, "status = conv2d_op.update(arguments, workspace.get()); \n");
    CutlassPrint(conv2d_decl, "CHECK(status == cutlass::Status::kSuccess);\n\n");
  }

  // Launch initialized CUTLASS kernel
  CutlassPrint(conv2d_decl, "status = conv2d_op();\n");
  CutlassPrint(conv2d_decl, "CHECK(status == cutlass::Status::kSuccess);\n\n");

  if (use_split_k) {
    CutlassPrint(conv2d_decl, "ReductionDevice reduction_op;\n");
    CutlassPrint(conv2d_decl,
                 "const static cutlass::conv::Operator kConvolutionalOperator = "
                 "Conv2d::kConvolutionalOperator;\n");
    CutlassPrint(conv2d_decl, "typename ReductionDevice::Arguments reduction_args(\n");
    CutlassPrint(conv2d_decl,
                 "cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, "
                 "problem_size).mn(),\n");
    CutlassPrint(conv2d_decl, "problem_size.split_k_slices,\n");
    CutlassPrint(conv2d_decl,
                 "cutlass::conv::implicit_gemm_tensor_c_size(kConvolutionalOperator, "
                 "problem_size),\n");
    CutlassPrint(conv2d_decl, "{\n");
    CutlassPrint(conv2d_decl,
                 " reinterpret_cast<Conv2d::ElementAccumulator*> (workspace.get()),\n");
    CutlassPrint(conv2d_decl,
                 "ReductionStrideIndex(tensor_c.stride()[Conv2d::ImplicitGemmKernel::"
                 "kTensorCStrideIdx])\n");
    CutlassPrint(conv2d_decl, "},\n");
    CutlassPrint(conv2d_decl, "{\n");
    CutlassPrint(conv2d_decl, "tensor_d.data(),\n");
    CutlassPrint(conv2d_decl,
                 "ReductionStrideIndex(tensor_d.stride()[Conv2d::ImplicitGemmKernel::"
                 "kTensorCStrideIdx])\n");
    CutlassPrint(conv2d_decl, "},\n");
    CutlassPrint(conv2d_decl, "{\n");
    CutlassPrint(conv2d_decl, "tensor_c.data(),\n");
    CutlassPrint(conv2d_decl,
                 "ReductionStrideIndex(tensor_c.stride()[Conv2d::ImplicitGemmKernel::"
                 "kTensorCStrideIdx])\n");
    CutlassPrint(conv2d_decl, "},\n");
    CutlassPrint(conv2d_decl, "   {alpha, beta}\n");
    CutlassPrint(conv2d_decl, ");\n\n");
    CutlassPrint(conv2d_decl, "status = reduction_op.initialize(reduction_args, nullptr);\n");
    CutlassPrint(conv2d_decl, "status = reduction_op();\n");
  }

  return conv2d_decl.str();
}

// class CodegenCutlass : public tvm::relax::backend::MemoizedExprTranslator<std::vector<Output>>,
//                        public CodegenCBase {
//  public:
// }

/*!
 * \brief Create a runtime module for CUTLASS.
 * \param ref The ext_func Relay expression/module to be executed using extern ops.
 * \return A runtime module.
 */
runtime::Module CUTLASSCompiler(const ObjectRef& ref) {
  ICHECK(ref->IsInstance<FunctionNode>()) << "The input ref is expected to be a Relax function.";
  Function func = Downcast<Function>(ref);
  std::string func_name = backend::GetExtSymbol(func);

  // const auto* pf = runtime::Registry::Get("runtime.CUTLASSJSONRuntimeCreate");
  // ICHECK(pf != nullptr) << "Cannot find CUTLASS runtime module create function.";
  // runtime::Module lib = (*pf)(func_name, graph_json, param_names);
  // return lib;
  return runtime::Module();
}

TVM_REGISTER_GLOBAL("relax.ext.cutlass").set_body_typed(CUTLASSCompiler);

}  // namespace contrib
}  // namespace relax
}  // namespace tvm
