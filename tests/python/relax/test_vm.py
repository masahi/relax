# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from __future__ import annotations  # must import to defer parsing of annotations
import pytest
import os
import numpy as np
import tvm
from tvm import relax, tir, te
from tvm.runtime import container
import numpy as np

from tvm.ir.base import assert_structural_equal
import tvm.script
from tvm.script import tir as T, relax as R
from tvm.relax.testing import nn


@tvm.register_func("test.vm.move")
def move(src):
    return src


@tvm.register_func("test.vm.add")
def add(a, b):
    ret = a.numpy() + b.numpy()
    return tvm.nd.array(ret)


@tvm.register_func("test.vm.mul")
def mul(a, b):
    ret = a.numpy() * b.numpy()
    return tvm.nd.array(ret)


@tvm.register_func("test.vm.identity")
def identity_packed(a, b):
    b[:] = tvm.nd.array(a.numpy())


@tvm.register_func("test.vm.tile")
def tile_packed(a, b):
    b[:] = tvm.nd.array(np.tile(a.numpy(), (1, 2)))


def test_vm_execute():
    ib = relax.ExecBuilder()
    with ib.function("func0", num_inputs=2):
        ib.emit_call("test.vm.add", args=[ib.r(0), ib.r(1)], dst=ib.r(2))
        ib.emit_ret(ib.r(2))
    ex = ib.get()
    vm = relax.VirtualMachine(ex, tvm.cpu())
    a = tvm.nd.array(
        np.random.rand(
            4,
        )
    )
    b = tvm.nd.array(
        np.random.rand(
            4,
        )
    )
    add_res = vm["func0"](a, b)
    np.testing.assert_allclose(add_res.numpy(), a.numpy() + b.numpy())


def test_vm_multiple_func():
    ib = relax.ExecBuilder()
    with ib.function("func0", num_inputs=2):
        ib.emit_call("test.vm.add", args=[ib.r(0), ib.r(1)], dst=ib.r(2))
        ib.emit_ret(ib.r(2))
    with ib.function("func1", num_inputs=2):
        ib.emit_call("test.vm.mul", args=[ib.r(0), ib.r(1)], dst=ib.r(2))
        ib.emit_ret(ib.r(2))
    ex = ib.get()
    vm = relax.VirtualMachine(ex, tvm.cpu())
    a = tvm.nd.array(
        np.random.rand(
            4,
        )
    )
    b = tvm.nd.array(
        np.random.rand(
            4,
        )
    )
    mul_res = vm["func1"](a, b)
    add_res = vm["func0"](a, b)
    np.testing.assert_allclose(add_res.numpy(), a.numpy() + b.numpy())
    np.testing.assert_allclose(mul_res.numpy(), a.numpy() * b.numpy())


def test_vm_serialize():
    ib = relax.ExecBuilder()
    with ib.function("func0", num_inputs=2):
        ib.emit_call("test.vm.add", args=[ib.r(0), ib.r(1)], dst=ib.r(2))
        ib.emit_ret(ib.r(2))
    arr = tvm.nd.array(
        np.random.rand(
            4,
        )
    )
    with ib.function("func1", num_inputs=2):
        ib.emit_call("test.vm.mul", args=[ib.r(0), arr], dst=ib.r(1))
        ib.emit_ret(ib.r(1))
    exec0 = ib.get()
    exec0.save_to_file("exec.tmp")
    exec1 = relax.load_exec_from_file("exec.tmp")
    assert exec0.astext() == exec1.astext()
    os.remove("exec.tmp")


def test_vm_constant_serialize():
    dtype = tvm.DataType("float32")
    shape = (4, 6)
    inp = tvm.nd.array(np.random.rand(4, 6).astype(np.float32))
    ib = relax.ExecBuilder()
    with ib.function("main", num_inputs=1):
        ib.emit_call(
            "vm.builtin.alloc_storage", args=[ib.vm_state(), (24,), ib.imm(1), dtype], dst=ib.r(1)
        )
        ib.emit_call(
            "vm.builtin.alloc_tensor", args=[ib.r(1), ib.imm(0), shape, dtype], dst=ib.r(2)
        )
        ib.emit_call("test.vm.identity", args=[ib.r(0), ib.r(2)])
        ib.emit_ret(ib.r(2))
    exec0 = ib.get()
    exec0.save_to_file("exec.tmp")
    exec1 = relax.load_exec_from_file("exec.tmp")
    assert exec0.astext() == exec1.astext()
    vm = relax.VirtualMachine(exec0, tvm.cpu())
    res = vm["main"](inp)
    np.testing.assert_allclose(inp.numpy(), res.numpy())
    os.remove("exec.tmp")


def test_vm_checker():
    ib = relax.ExecBuilder()
    try:
        with ib.function("func0", num_inputs=2):
            ib.emit_call("test.vm.add", args=[ib.r(0), ib.r(2)], dst=ib.r(2))
            ib.emit_ret(ib.r(2))
        ib.get()
    except ValueError as ex:
        assert True


def test_vm_formalize():
    ib0 = relax.ExecBuilder()
    ib1 = relax.ExecBuilder()
    with ib0.function("func0", num_inputs=2):
        ib0.emit_call("test.vm.add", args=[ib0.r(0), ib0.r(1)], dst=ib0.r(100))
        ib0.emit_call("test.vm.mul", args=[ib0.r(1), ib0.r(100)], dst=ib0.r(50))
        ib0.emit_ret(ib0.r(50))
    with ib1.function("func0", num_inputs=2):
        ib1.emit_call("test.vm.add", args=[ib1.r(0), ib1.r(1)], dst=ib1.r(2))
        ib1.emit_call("test.vm.mul", args=[ib1.r(1), ib1.r(2)], dst=ib1.r(3))
        ib1.emit_ret(ib1.r(3))
    exec0 = ib0.get()
    exec1 = ib1.get()
    assert exec0.astext() == exec1.astext()


@tvm.register_func("test.vm.add_scalar")
def add_scalar(a, b):
    return a + b


@tvm.register_func("test.vm.get_device_id")
def get_device_id(device):
    return device.device_id


def test_vm_operand():
    ib0 = relax.ExecBuilder()
    with ib0.function("func0", num_inputs=2):
        ib0.emit_call("test.vm.add_scalar", args=[ib0.r(0), ib0.r(1)], dst=ib0.r(2))
        ib0.emit_ret(ib0.r(2))
    exec0 = ib0.get()
    vm = relax.VirtualMachine(exec0, tvm.cpu())
    res = vm["func0"](2, 3)
    assert res == 5

    ib1 = relax.ExecBuilder()
    with ib1.function("func1", num_inputs=1):
        ib1.emit_call("test.vm.get_device_id", args=[ib1.r(0)], dst=ib1.r(1))
        ib1.emit_ret(ib1.r(1))
    exec1 = ib1.get()
    vm = relax.VirtualMachine(exec1, tvm.cpu())
    res = vm["func1"](tvm.cpu(3))
    assert res == 3


def test_vm_shapeof():
    ib = relax.ExecBuilder()
    shape = (32, 16)
    arr = tvm.nd.array(np.random.rand(*shape))
    with ib.function("main", num_inputs=0):
        ib.emit_call("vm.builtin.shape_of", args=[arr], dst=ib.r(0))
        ib.emit_ret(ib.r(0))
    ex = ib.get()
    vm = relax.VirtualMachine(ex, tvm.cpu())
    res = vm["main"]()
    for i, s in enumerate(res):
        assert s == shape[i]


def test_vm_storage():
    dtype = tvm.DataType("float32")
    shape = (4, 6)
    ib = relax.ExecBuilder()
    with ib.function("main", num_inputs=0):
        ib.emit_call(
            "vm.builtin.alloc_storage", args=[ib.vm_state(), (24,), ib.imm(1), dtype], dst=ib.r(1)
        )
        ib.emit_call(
            "vm.builtin.alloc_tensor", args=[ib.r(1), ib.imm(0), shape, dtype], dst=ib.r(2)
        )
        ib.emit_ret(ib.r(2))
    ex = ib.get()
    vm = relax.VirtualMachine(ex, tvm.cpu())
    res = vm["main"]()
    assert res.device == tvm.cpu()
    assert res.shape == shape


def test_vm_copy():
    @tvm.script.ir_module
    class TestVMMove:
        @R.function
        def foo(x: Tensor[(3, 4), "float32"]):
            z = R.call_packed("vm.builtin.copy", x)
            return z

    mod = TestVMMove
    target = tvm.target.Target("llvm", host="llvm")
    ex, lib = relax.vm.build(mod, target)
    inp = tvm.nd.array(np.random.rand(3, 4).astype(np.float32))
    vm = relax.VirtualMachine(ex, tvm.cpu(), mod=lib)
    res = vm["foo"](inp)
    np.testing.assert_allclose(res.numpy(), inp.numpy())


def test_vm_goto():
    ib = relax.ExecBuilder()
    with ib.function("main", num_inputs=2):
        ib.emit_call("test.vm.add", args=[ib.r(0), ib.r(1)], dst=ib.r(2))
        ib.emit_goto(2)
        ib.emit_call("test.vm.mul", args=[ib.r(2), ib.r(1)], dst=ib.r(2))
        ib.emit_ret(ib.r(2))
    ex = ib.get()
    vm = relax.VirtualMachine(ex, tvm.cpu())
    a = tvm.nd.array(
        np.random.rand(
            4,
        )
    )
    b = tvm.nd.array(
        np.random.rand(
            4,
        )
    )
    res = vm["main"](a, b)
    np.testing.assert_allclose(res.numpy(), a.numpy() + b.numpy())


def test_vm_if():
    ib = relax.ExecBuilder()
    with ib.function("main", num_inputs=3):
        ib.emit_if(ib.r(0), 3)
        ib.emit_call("test.vm.add", args=[ib.r(1), ib.r(2)], dst=ib.r(3))
        ib.emit_goto(2)
        ib.emit_call("test.vm.mul", args=[ib.r(1), ib.r(2)], dst=ib.r(3))
        ib.emit_ret(ib.r(3))
    ex = ib.get()
    vm = relax.VirtualMachine(ex, tvm.cpu())
    a = tvm.nd.array(
        np.random.rand(
            4,
        )
    )
    b = tvm.nd.array(
        np.random.rand(
            4,
        )
    )
    res = vm["main"](False, a, b)
    np.testing.assert_allclose(res.numpy(), a.numpy() * b.numpy())
    res = vm["main"](1, a, b)
    np.testing.assert_allclose(res.numpy(), a.numpy() + b.numpy())


def test_vm_compile_if():
    @tvm.script.ir_module
    class TestVMCompileIf:
        @R.function
        def ife(cond: Tensor[(), "bool"], x: Tensor[(3, 4), "float32"]):
            if cond:
                w = relax.call_packed("test.vm.add", x, x)
            else:
                w = relax.call_packed("test.vm.mul", x, x)
            return w

    mod = TestVMCompileIf
    target = tvm.target.Target("llvm", host="llvm")
    ex, lib = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu(), mod=lib)
    inp = tvm.nd.array(np.random.rand(3, 4))
    res = vm["ife"](True, inp)
    np.testing.assert_allclose(res.numpy(), inp.numpy() + inp.numpy())
    res = vm["ife"](0, inp)
    np.testing.assert_allclose(res.numpy(), inp.numpy() * inp.numpy())


def test_vm_compile_stage0():
    @tvm.script.ir_module
    class TestVMCompileStage0:
        @R.function
        def foo(x: Tensor[(3, 4), "float32"], y: Tensor[(3, 4), "float32"]):
            z = R.call_packed("test.vm.identity", x, y)
            return y

    mod = TestVMCompileStage0
    target = tvm.target.Target("llvm", host="llvm")
    ex, lib = relax.vm.build(mod, target)
    inp1 = tvm.nd.array(np.random.rand(3, 4).astype(np.float32))
    inp2 = tvm.nd.array(np.random.rand(3, 4).astype(np.float32))
    vm = relax.VirtualMachine(ex, tvm.cpu(), mod=lib)
    vm["foo"](inp1, inp2)
    np.testing.assert_allclose(inp2.numpy(), inp1.numpy())


def test_vm_compile_stage1():
    @tvm.script.ir_module
    class TestVMCompileStage1:
        @T.prim_func
        def shape_func0(heap: T.handle) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "shape_func0"})
            H = T.match_buffer(
                heap,
                [T.int64(4)],
                dtype="int64",
                elem_offset=T.int64(0),
                align=128,
                offset_factor=1,
            )
            # body
            T.store(H.data, T.int64(2), (T.load("int64", H.data, T.int64(0)) * T.int64(2)), True)
            T.store(H.data, T.int64(3), (T.load("int64", H.data, T.int64(1)) * T.int64(3)), True)

        @R.function
        def foo(x: Tensor[_, "float32"]) -> Shape:
            shape_heap: Tensor[(4,), "int64"] = relax.call_packed(
                "vm.builtin.alloc_shape_heap", (4,)
            )
            gv0 = relax.call_packed("vm.builtin.shape_of", x)
            gv1 = relax.call_packed("vm.builtin.store_shape", gv0, shape_heap, (0, 1))
            gv2 = shape_func0(shape_heap)
            gv3 = relax.call_packed("vm.builtin.load_shape", shape_heap, (2, 3))
            return gv3

    mod = TestVMCompileStage1
    target = tvm.target.Target("llvm", host="llvm")
    ex, lib = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu(), mod=lib)

    shape = (32, 16)
    arr = tvm.nd.array(np.random.rand(*shape))
    res = vm["foo"](arr)
    assert res[0] == shape[0] * 2
    assert res[1] == shape[1] * 3


def test_vm_compile_stage2():
    @tvm.script.ir_module
    class TestVMCompileStage2:
        @R.function
        def foo(x: Tensor[_, "float32"]) -> Shape:
            R.match_shape(x, (n, m))
            return (n * 2, m * 3)

    mod = TestVMCompileStage2
    target = tvm.target.Target("llvm", host="llvm")
    ex, lib = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu(), mod=lib)

    shape = (32, 16)
    arr = tvm.nd.array(np.random.rand(*shape))
    res = vm["foo"](arr)
    assert res[0] == shape[0] * 2
    assert res[1] == shape[1] * 3


def test_vm_compile_stage3():
    @tvm.script.ir_module
    class TestVMCompileStage3:
        @R.function
        def foo(x: Tensor[(32, 16), "float32"]) -> Tensor:
            with R.dataflow():
                y = R.call_tir((32, 16), "test.vm.identity", (x))
                R.output(y)
            return y

    mod = TestVMCompileStage3
    target = tvm.target.Target("llvm", host="llvm")
    ex, lib = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu(), mod=lib)

    shape = (32, 16)
    inp = tvm.nd.array(np.random.rand(*shape).astype(np.float32))
    res = vm["foo"](inp)
    np.testing.assert_allclose(inp.numpy(), res.numpy())


def test_vm_compile_e2e():
    @tvm.script.ir_module
    class TestVMCompileE2E:
        @R.function
        def foo(x: Tensor[_, "float32"]) -> Tensor:
            with R.dataflow():
                R.match_shape(x, (n, m))
                y = R.call_tir((n, m * 2), "test.vm.tile", (x))
                R.output(y)
            return y

    mod = TestVMCompileE2E

    target = tvm.target.Target("llvm", host="llvm")
    ex, lib = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu(), mod=lib)

    shape = (32, 16)
    inp = tvm.nd.array(np.random.rand(*shape).astype(np.float32))
    res = vm["foo"](inp)
    np.testing.assert_allclose(np.tile(inp.numpy(), (1, 2)), res.numpy())


def test_vm_compile_e2e_func_param_with_shape():
    @tvm.script.ir_module
    class TestVMCompileE2E2:
        @T.prim_func
        def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
            T.func_attr({"global_symbol": "tir_matmul"})
            m = T.var("int32")
            n = T.var("int32")
            k = T.var("int32")
            A = T.match_buffer(x, (m, n))
            B = T.match_buffer(y, (n, k))
            C = T.match_buffer(z, (m, k))

            for i, j, k in T.grid(m, k, n):
                with T.block("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        @R.function
        def func(x: Tensor[(m, n), "float32"], w: Tensor[(n, k), "float32"]) -> Tensor:
            gv0 = R.call_tir((m, k), tir_matmul, (x, w))
            return gv0

    mod = TestVMCompileE2E2

    target = tvm.target.Target("llvm", host="llvm")
    ex, lib = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu(), mod=lib)

    data = tvm.nd.array(np.random.rand(32, 16).astype(np.float32))
    weight = tvm.nd.array(np.random.rand(16, 32).astype(np.float32))
    res = vm["func"](data, weight)
    expected = np.dot(data.numpy(), weight.numpy())
    np.testing.assert_allclose(expected, res.numpy(), rtol=1e-4, atol=1e-4)


def test_vm_emit_te_extern():
    if not tvm.get_global_func("tvm.contrib.cblas.matmul", True):
        print("skip because extern function is not available")
        return
    bb = relax.BlockBuilder()
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    type_anno = relax.DynTensorType(2, "float32")
    x = relax.Var("x", [n, m], type_anno)
    y = relax.Var("y", [m, n], type_anno)

    with bb.function("rx_cblas_matmul", [x, y]):
        out = bb.emit_te(tvm.contrib.cblas.matmul, x, y, transa=False, transb=False)
        bb.emit_func_output(out)

    mod = bb.get()

    target = tvm.target.Target("llvm", host="llvm")
    ex, lib = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu(), mod=lib)

    data = tvm.nd.array(np.random.rand(16, 32).astype(np.float32))
    weight = tvm.nd.array(np.random.rand(32, 16).astype(np.float32))
    res = vm["rx_cblas_matmul"](data, weight)
    expected = np.dot(data.numpy(), weight.numpy())
    np.testing.assert_allclose(expected, res.numpy(), rtol=1e-4, atol=1e-4)


def test_vm_emit_te_concat():
    # concatenate of two vectors of size (n,) and (m,)
    bb = relax.BlockBuilder()
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    type_anno = relax.DynTensorType(1, "float32")
    x = relax.Var("x", [n], type_anno)
    y = relax.Var("y", [m], type_anno)

    def te_func(A, B):
        C = te.compute((n + m), lambda i: tvm.tir.if_then_else(i < n, A[i], B[i - n]))
        return C

    with bb.function("rx_func", [x, y]):
        x1 = bb.emit_te(te_func, x, y)
        bb.emit_func_output(x1)

    mod = bb.get()

    target = tvm.target.Target("llvm", host="llvm")
    ex, lib = relax.vm.build(mod, target)

    vm = relax.VirtualMachine(ex, tvm.cpu(), mod=lib)
    inp = tvm.nd.array(
        np.random.rand(
            1,
        ).astype(np.float32)
    )
    inp2 = tvm.nd.array(
        np.random.rand(
            2,
        ).astype(np.float32)
    )
    res = vm["rx_func"](inp, inp2)

    np.testing.assert_allclose(res.numpy(), np.append(inp.numpy(), inp2.numpy()))


def test_vm_emit_te_floor_symbolic_shape():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    type_anno = relax.DynTensorType(1, "float32")
    x = relax.Var("x", [n], type_anno)

    def te_func(A):
        C = te.compute((tir.floordiv(n, 2),), lambda i: A[i] + 1)
        return C

    with bb.function("rx_func", [x]):
        x1 = bb.emit_te(te_func, x)
        bb.emit_func_output(x1)

    mod = bb.get()

    target = tvm.target.Target("llvm", host="llvm")
    ex, lib = relax.vm.build(mod, target)

    vm = relax.VirtualMachine(ex, tvm.cpu(), mod=lib)
    shape = (9,)
    inp = tvm.nd.array(np.random.rand(*shape).astype(np.float32))
    res = vm["rx_func"](inp)

    def expected_output():
        output_shape = (shape[0] // 2,)
        return inp.numpy()[: output_shape[0]] + 1

    np.testing.assert_allclose(res.numpy(), expected_output())


def test_vm_relax_symbolic_shape():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    type_anno = relax.DynTensorType(1, "float32")
    x = relax.Var("x", [n], type_anno)
    y = relax.Var("y", [(n // 2) + 1], type_anno)

    def te_func(A, B):
        C = te.compute((n,), lambda i: A[i] + B[i // 2])
        return C

    with bb.function("rx_func", [x, y]):
        x1 = bb.emit_te(te_func, x, y)
        bb.emit_func_output(x1)

    mod = bb.get()

    target = tvm.target.Target("llvm", host="llvm")
    ex, lib = relax.vm.build(mod, target)

    vm = relax.VirtualMachine(ex, tvm.cpu(), mod=lib)
    shape1 = (5,)
    shape2 = (3,)
    inp = tvm.nd.array(np.random.rand(*shape1).astype(np.float32))
    inp2 = tvm.nd.array(np.random.rand(*shape2).astype(np.float32))
    res = vm["rx_func"](inp, inp2)

    def expected_output():
        return inp.numpy() + np.repeat(inp2.numpy(), 2)[:5]

    np.testing.assert_allclose(res.numpy(), expected_output())


def test_vm_relax_dyn_tir_shape():
    # case where TIR variables are unbound in generated PrimFunc
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")

    def te_func(A):
        C = te.compute((n + 1), lambda i: A[i])
        return C

    with bb.function("rx_func"):
        x = nn.Placeholder((n,), dtype="float32", name="x")
        y = nn.Placeholder((n + 1,), dtype="float32", name="y")

        x1 = bb.emit_te(te_func, y)
        bb.emit_func_output(x1, params=[x, y])

    mod = bb.get()

    target = tvm.target.Target("llvm", host="llvm")
    ex, lib = relax.vm.build(mod, target)

    ex.save_to_file("exec.tmp")
    exec1 = relax.load_exec_from_file("exec.tmp")
    assert ex.astext() == exec1.astext()

    vm = relax.VirtualMachine(ex, tvm.cpu(), mod=lib)
    inp = tvm.nd.array(np.random.rand(2).astype(np.float32))
    inp2 = tvm.nd.array(np.random.rand(3).astype(np.float32))

    res = vm["rx_func"](inp, inp2)

    np.testing.assert_allclose(res.numpy(), inp2.numpy())
    os.remove("exec.tmp")


def test_vm_tuple():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")

    with bb.function("rx_func"):
        x = nn.Placeholder((n,), dtype="float32", name="x")
        y = nn.Placeholder((n,), dtype="float32", name="y")
        tup = relax.Tuple([x, y])
        item = tup[0]
        bb.emit_func_output([tup, item], params=[x, y])

    mod = bb.get()

    target = tvm.target.Target("llvm", host="llvm")
    ex, lib = relax.vm.build(mod, target)

    vm = relax.VirtualMachine(ex, tvm.cpu(), mod=lib)
    shape = (5, 5)
    inp = tvm.nd.array(np.random.rand(*shape).astype(np.float32))
    inp2 = tvm.nd.array(np.random.rand(*shape).astype(np.float32))
    (res1, res2), res3 = vm["rx_func"](inp, inp2)

    np.testing.assert_allclose(res1.numpy(), inp.numpy())
    np.testing.assert_allclose(res2.numpy(), inp2.numpy())
    np.testing.assert_allclose(res3.numpy(), inp.numpy())


def test_vm_tuplegetitem():
    @tvm.script.ir_module
    class TestVMTupleGetItem:
        @R.function
        def tuple_get_item(x: Tensor[(_, _), "float32"], y: Tensor[(_, _), "float32"]):
            t = relax.Tuple((x, y))
            a = relax.TupleGetItem(t, 0)
            b = relax.TupleGetItem(t, 1)
            c = relax.call_packed("test.vm.add", a, b)
            return c

    mod = TestVMTupleGetItem
    target = tvm.target.Target("llvm", host="llvm")
    ex, lib = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu(), mod=lib)
    x_inp = tvm.nd.array(np.random.rand(2, 3))
    y_inp = tvm.nd.array(np.random.rand(2, 3))
    res = vm["tuple_get_item"](x_inp, y_inp)
    np.testing.assert_allclose(res.numpy(), x_inp.numpy() + y_inp.numpy())


if __name__ == "__main__":
    pytest.main([__file__])
