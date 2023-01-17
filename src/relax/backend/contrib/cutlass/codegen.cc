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
#include "../../../../relay/backend/contrib/cutlass/codegen.h"

#include <tvm/ir/module.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/nn.h>
#include <tvm/relax/type.h>

#include <memory>
#include <string>
#include <vector>

#include "../../../../relay/backend/contrib/codegen_c/codegen_c.h"
#include "../utils.h"

namespace tvm {
namespace relax {
namespace contrib {

using namespace relay::contrib::cutlass;
using Output = relay::contrib::Output;
using GenerateBodyOutput = relay::contrib::GenerateBodyOutput;
using relay::contrib::cutlass::GenerateBody;

class CodegenCutlass : public tvm::relax::backend::MemoizedExprTranslator<std::vector<Output>>,
                       public relay::contrib::CodegenCBase {
 public:
  CodegenCutlass(const std::string& id, const Map<String, ObjectRef>& attrs, const Expr& func_expr)
      : ext_func_id_(id), attrs_(attrs), bindings_(AnalyzeVar2Value(func_expr)) {}

  std::string JIT(const std::vector<Output>& out) final {
    std::vector<std::string> arg_types, arg_names;

    for (const auto& arg : ext_func_args_) {
      auto sinfo = GetStructInfo(arg);
      if (const auto* tensor_sinfo = sinfo.as<TensorStructInfoNode>()) {
        arg_types.emplace_back(backend::DType2String(tensor_sinfo->dtype));
      } else {
        LOG(FATAL) << "Unimplemented";
      }
      arg_names.push_back(arg->name_hint());
    }

    code_stream_ << EmitSignature(out, ext_func_id_, arg_names) << "{\n";

    this->EnterScope();

    // Function body
    for (auto decl : buf_decl_) {
      this->PrintIndents();
      code_stream_ << decl << "\n";
    }
    code_stream_ << "\n";
    for (auto stmt : ext_func_body_) {
      this->PrintIndents();
      code_stream_ << stmt << "\n";
    }

    this->ExitScope();
    code_stream_ << "}\n";

    this->GenerateBackendCFunc(ext_func_id_, arg_types, /*const_arr_name=*/"", out, true);
    return code_stream_.str();
  }

 protected:
  std::vector<Output> VisitExpr_(const VarNode* node) final {
    ext_func_args_.push_back(GetRef<Var>(node));
    Output output;
    output.name = node->name_hint();
    return {output};
  }

  std::vector<Output> VisitExpr_(const CallNode* call) final {
    const auto* fn_var = call->op.as<VarNode>();
    ICHECK(fn_var);
    const auto func = Downcast<Function>(bindings_[GetRef<Var>(fn_var)]);
    ICHECK(func.defined()) << "Only composite function is supported for CUTLASS.";
    GenerateBodyOutput ret = GenerateCompositeFunctionCall(func, call);
    ext_func_body_.push_back(ret.decl);
    return ret.outputs;
  }

  std::vector<Output> VisitExpr_(const FunctionNode* fn) {
    ICHECK(fn->GetAttr<String>(attr::kComposite).defined())
        << "JSON runtime only supports composite functions";
    // FunctionNode should be handled by the caller.
    return {};
  }

  std::vector<Output> VisitBinding_(const VarBindingNode* binding) {
    ICHECK_EQ(memo_.count(binding->var), 0);
    memo_[binding->var] = VisitExpr(binding->value);
    return VisitExpr(binding->value);
  }

  std::vector<Output> VisitBinding(const Binding& binding) {
    std::vector<Output> outputs;
    if (const auto* node = binding.as<VarBindingNode>()) {
      auto from_b = VisitBinding_(node);
      outputs.insert(outputs.end(), from_b.begin(), from_b.end());
    } else {
      LOG(FATAL) << "Unimplemented type: " << binding->GetTypeKey();
    }
    return outputs;
  }

  std::vector<Output> VisitBindingBlock(const BindingBlock& block) {
    std::vector<Output> outputs;
    if (const auto* node = block.as<DataflowBlockNode>()) {
      auto from_bb = VisitBindingBlock_(node);
      outputs.insert(outputs.end(), from_bb.begin(), from_bb.end());
    } else if (const auto* node = block.as<BindingBlockNode>()) {
      auto from_bb = VisitBindingBlock_(node);
      outputs.insert(outputs.end(), from_bb.begin(), from_bb.end());
    } else {
      LOG(FATAL) << "TypeError: Invalid type: " << block->GetTypeKey();
    }
    return outputs;
  }

  std::vector<Output> VisitBindingBlock_(const BindingBlockNode* block) {
    std::vector<Output> outputs;
    for (Binding binding : block->bindings) {
      auto from_b = VisitBinding(binding);
      outputs.insert(outputs.end(), from_b.begin(), from_b.end());
    }
    return outputs;
  }

  std::vector<Output> VisitBindingBlock_(const DataflowBlockNode* block) {
    std::vector<Output> outputs;
    for (Binding binding : block->bindings) {
      auto from_b = VisitBinding(binding);
      outputs.insert(outputs.end(), from_b.begin(), from_b.end());
    }
    return outputs;
  }

  std::vector<Output> VisitExpr_(const SeqExprNode* op) {
    std::vector<Output> outputs;

    for (BindingBlock block : op->blocks) {
      auto from_bb = VisitBindingBlock(block);
      // TODO: why comment out?
      // outputs.insert(outputs.end(), from_bb.begin(), from_bb.end());
    }

    auto from_body = VisitExpr(op->body);
    outputs.insert(outputs.end(), from_body.begin(), from_body.end());

    return outputs;
  }

 private:
  std::vector<std::string> GetArgumentNames(const CallNode* call) {
    std::vector<std::string> arg_names;
    for (size_t i = 0; i < call->args.size(); ++i) {
      auto res = VisitExpr(call->args[i]);
      for (const auto& out : res) {
        arg_names.push_back(out.name);
      }
    }
    return arg_names;
  }

  GenerateBodyOutput GenerateCompositeFunctionCall(Function callee, const CallNode* caller) {
    const auto pattern_name = callee->GetAttr<runtime::String>(attr::kComposite);
    ICHECK(pattern_name.defined()) << "Only functions with composite attribute are supported.";

    if (pattern_name == "conv2d_bias_relu") {
      const CallNode* conv2d_call = caller;
      for (auto [var, val] : bindings_) {
        if (val->IsInstance<CallNode>() && backend::IsOp(val.as<CallNode>(), "relax.nn.conv2d")) {
          conv2d_call = val.as<CallNode>();
          break;
        }
      }
      return GenerateBody(conv2d_call, "cutlass_conv2d_bias_relu", GetArgumentNames(caller),
                          Conv2dArgs(std::ref(attrs_)));
    }

    LOG(FATAL) << "Unknown composite function: " << pattern_name;
    return {};
  }

  GenerateBodyOutput GenerateBody(const CallNode* root_call, const std::string& func_name,
                                  const std::vector<std::string>& func_args,
                                  const Str2StrMap& attribute_args) {
    auto struct_info = GetStructInfo(GetRef<Call>(root_call));

    std::vector<std::string> out_types;
    if (const auto* tensor_sinfo = struct_info.as<TensorStructInfoNode>()) {
      out_types.emplace_back(backend::DType2String(tensor_sinfo->dtype));
    } else {
      LOG(FATAL) << "Unimplemented";
    }

    return contrib::GenerateBody(func_name, ext_func_id_, func_args, out_types, attribute_args,
                                 &buf_idx_);
  }

  /*! \brief The id of the external cutlass ext_func. */
  std::string ext_func_id_;
  /*! \brief The attrs of the external cutlass ext_func. */
  Map<String, ObjectRef> attrs_;
  /*!
   * \brief The index to track the output buffer. Each kernel will redirect the
   * output to a buffer that may be consumed by other kernels.
   */
  int buf_idx_{0};
  /*! \brief The arguments used by a wrapped function that calls CUTLASS kernels. */
  Array<Var> ext_func_args_;
  /*! \brief Statement of the function that will be compiled using CUTLASS kernels. */
  std::vector<std::string> ext_func_body_;
  /*! \brief The declaration of intermediate buffers. */
  std::vector<std::string> buf_decl_;

  Map<Var, Expr> bindings_;
};

class CutlassModuleCodegen {
 public:
  runtime::Module CreateCSourceModule(Function f) {
    auto code = EmitHeaders() + GenCutlassFunc(f);
    return Finalize(code, func_names_);
  }

 private:
  std::string GenCutlassFunc(const Function& function) {
    ICHECK(function.defined()) << "Input error: expect a Relay function.";

    // Record the external symbol for runtime lookup.
    Optional<String> opt_global_symbol = function->GetAttr<String>(tvm::attr::kGlobalSymbol);
    ICHECK(opt_global_symbol.defined())
        << "CUTLASS functions must have a " << tvm::attr::kGlobalSymbol << " attribute";
    std::string sid = opt_global_symbol.value();
    func_names_.push_back(sid);

    const auto* attrs = function->attrs.as<DictAttrsNode>();
    ICHECK(attrs != nullptr);
    CodegenCutlass builder(sid, attrs->dict, function);
    auto out = builder.VisitExpr(function->body);
    return builder.JIT(out);
  }

  /*! \brief The accumulated function names. */
  Array<String> func_names_;
};  // CutlassModuleCodegen

/*!
 * \brief Create a runtime module for CUTLASS.
 * \param ref The ext_func Relay expression/module to be executed using extern ops.
 * \return A runtime module.
 */
runtime::Module CUTLASSCompiler(const ObjectRef& ref) {
  ICHECK(ref->IsInstance<FunctionNode>()) << "The input ref is expected to be a Relax function.";
  Function func = Downcast<Function>(ref);
  std::string func_name = backend::GetExtSymbol(func);
  auto source_mod = CutlassModuleCodegen().CreateCSourceModule(func);
  const auto* pf = runtime::Registry::Get("contrib.cutlass.compile");
  ICHECK(pf != nullptr);
  return (*pf)(source_mod);
}

TVM_REGISTER_GLOBAL("relax.ext.cutlass").set_body_typed(CUTLASSCompiler);

}  // namespace contrib
}  // namespace relax
}  // namespace tvm
