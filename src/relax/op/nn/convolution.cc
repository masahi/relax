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
 * \file src/relax/op/nn/convolution.cc
 * \brief Convolution operators
 */

#include "../op_common.h"

namespace tvm {
namespace relax {

TVM_REGISTER_NODE_TYPE(Conv2DAttrs);

template <typename T>
inline Expr MakeConv(Expr data, Expr weight, Array<PrimExpr> strides, Array<PrimExpr> padding,
                     Array<PrimExpr> dilation, int groups, PrimExpr channels,
                     Array<PrimExpr> kernel_size, std::string data_layout,
                     std::string kernel_layout, std::string out_layout, DataType out_dtype,
                     std::string op_name) {
  auto attrs = make_object<T>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->channels = std::move(channels);
  attrs->kernel_size = std::move(kernel_size);
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = std::move(out_layout);
  attrs->out_dtype = std::move(out_dtype);
  const Op& op = Op::Get(op_name);
  return Call(op, {data, weight}, Attrs(attrs), {});
}

Expr MakeConv2D(Expr data, Expr weight, Array<PrimExpr> strides, Array<PrimExpr> padding,
                Array<PrimExpr> dilation, int groups, PrimExpr channels,
                Array<PrimExpr> kernel_size, String data_layout, String kernel_layout,
                String out_layout, DataType out_dtype) {
  return MakeConv<Conv2DAttrs>(data, weight, strides, padding, dilation, groups, channels,
                               kernel_size, data_layout, kernel_layout, out_layout, out_dtype,
                               "relax.nn.conv2d");
}

StructInfo InferStructInfoConv2d(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    ctx->ReportFatal(Diagnostic::Error(call->span) << "Conv2d op should have 2 arguments");
  }

  auto* data_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  auto* weight_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[1]);

  if (!data_sinfo || !weight_sinfo) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Both lhs and rhs should be Tensor for broadcasting, but got "
                     << call->args[0]->struct_info_->GetTypeKey() << " and "
                     << call->args[0]->struct_info_->GetTypeKey());
  }

  DataType output_dtype;
  if (data_sinfo->IsUnknownDtype() || weight_sinfo->IsUnknownDtype()) {
    output_dtype = DataType::Void();
  } else if (data_sinfo->dtype != weight_sinfo->dtype) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Data types " << data_sinfo->dtype << " and " << weight_sinfo->dtype
                     << " must be equal for broadcasting operators");
  } else {
    output_dtype = data_sinfo->dtype;
  }

  // ndims
  int output_ndim;
  if (data_sinfo->IsUnknownNdim() || weight_sinfo->IsUnknownNdim()) {
    output_ndim = kUnknownNDim;
  } else {
    output_ndim = std::max(data_sinfo->ndim, weight_sinfo->ndim);
  }

  auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  auto* weight_shape = weight_sinfo->shape.as<ShapeExprNode>();
  auto* attrs = call->attrs.as<Conv2DAttrs>();

  if (data_shape && weight_shape) {
    std::vector<PrimExpr> output_shape;
    size_t ndim0 = data_shape->values.size();
    size_t ndim1 = weight_shape->values.size();
    if (ndim0 != 4 || ndim1 != 4) {
      LOG(INFO) << ndim0;
      LOG(INFO) << ndim1;
      ctx->ReportFatal(Diagnostic::Error(call) << "The 2 arguments of Conv2d must be 4D Tensors");
    }
    // N
    output_shape.push_back(data_shape->values[0]);
    // C
    output_shape.push_back(weight_shape->values[0]);
    // H
    output_shape.push_back((data_shape->values[2] + 2 * attrs->padding[0] -
                            attrs->dilation[0] * (attrs->kernel_size[0] - 1) - 1) /
                               attrs->strides[0] +
                           1);
    // W
    output_shape.push_back((data_shape->values[3] + 2 * attrs->padding[1] -
                            attrs->dilation[1] * (attrs->kernel_size[1] - 1) - 1) /
                               attrs->strides[1] +
                           1);
    ShapeExpr shape(Array<PrimExpr>{output_shape.begin(), output_shape.end()});
    return TensorStructInfo(shape, output_dtype);
  }
  return TensorStructInfo(output_dtype, /*ndim=*/output_ndim);
}

TVM_REGISTER_GLOBAL("relax.op.nn.conv2d").set_body_typed(MakeConv2D);

RELAY_REGISTER_OP("relax.nn.conv2d")
    .describe(R"code(2D convolution layer (e.g. spatial convolution over images).

This layer creates a convolution kernel that is convolved
with the layer input to produce a tensor of outputs.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, in_channels, height, width) if `layout` is `NCHW`.
- **weight**: (channels, in_channels, kernel_size[0], kernel_size[1])
- **out**:  This depends on the `layout` parameter. Output is 4D array of shape
            (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.

)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_attrs_type<Conv2DAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoConv2d);

TVM_REGISTER_GLOBAL("relax.op.nn.relu").set_body_typed([](Expr inp) {
  const Op& op = Op::Get("relax.nn.relu");
  return Call(op, {inp});
});

RELAY_REGISTER_OP("relax.nn.relu")
    .describe(
        "This operation returns the unique elements and the new index of each item in a given "
        "tensor.")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoUnary<false>);

}  // namespace relax
}  // namespace tvm
