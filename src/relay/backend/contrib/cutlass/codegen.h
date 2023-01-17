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
 * \file src/relay/backend/contrib/cutlass/codegen.h
 * \brief The 'custom' compilation pass for CUTLASS (invoked by the RelayToTIRTargetHook pass).
 */

#ifndef TVM_RELAY_BACKEND_CONTRIB_CUTLASS_CODEGEN_H_
#define TVM_RELAY_BACKEND_CONTRIB_CUTLASS_CODEGEN_H_

#include <tvm/ir/transform.h>

#include "../codegen_c/codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace cutlass {

using Str2StrMap = std::unordered_map<std::string, std::string>;

inline void CutlassPrint(std::ostringstream& os, const std::string& stmt, int indent = 2) {
  for (int i = 0; i < indent; ++i) {
    os << " ";
  }
  os << stmt;
}

std::string GetDimAsStr(ObjectRef dim);

Str2StrMap ArgsCommon(const Map<String, ObjectRef>& attrs);

Str2StrMap GemmArgsCommon(const Map<String, ObjectRef>& attrs);

Str2StrMap DenseArgs(const Map<String, ObjectRef>& attrs);

Str2StrMap BatchMatmulArgs(const Map<String, ObjectRef>& attrs);

void AppendPrologue(std::ostringstream& gemm_decl, const Str2StrMap& attrs,
                    const std::vector<std::string>& func_args, const std::string& kernel,
                    bool has_bias, bool is_gelu, int m_axis_idx, int n_axis_idx, int k_axis_idx);

void AppendGemmExecute(std::ostringstream& gemm_decl, const std::string& kernel);

std::string DenseOp(std::string id, const Str2StrMap& attrs,
                    const std::vector<std::string>& func_args);

std::string BatchMatmulOp(std::string id, const Str2StrMap& attrs,
                          const std::vector<std::string>& func_args);

Str2StrMap Conv2dArgs(const Map<String, ObjectRef>& attrs, bool is_dgrad = false,
                      bool is_wgrad = false);

std::string Conv2dOp(std::string id, const Str2StrMap& attrs,
                     const std::vector<std::string>& func_args, bool has_residual_block = false);

std::string EmitHeaders();

std::string EmitSignature(const std::vector<relay::contrib::Output>& out,
                          const std::string& func_id, const std::vector<std::string>& arg_names);

GenerateBodyOutput GenerateBody(const std::string& func_name, const std::string& ext_func_id,
                                const std::vector<std::string>& func_args,
                                const std::vector<std::string>& output_types,
                                const Str2StrMap& attribute_args, int* buf_idx);

runtime::Module Finalize(const std::string& code, const Array<String>& func_names);
/*!
 * \brief Returns the pass which replaces all calls to "Primitive" functions with "Compiler"
 * attribute of "cutlass" with an call to an extern, and binds a \p runtime::StaticLibrary
 * to the IRModule's "external_mods" attribute containing compiled implementations of
 * those functions using the CUTLASS C++ template library.
 */
transform::Pass CompileForCutlass();

}  // namespace cutlass
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_CUTLASS_CODEGEN_H_
