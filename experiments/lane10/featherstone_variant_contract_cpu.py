#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only AST source-contract check for Newton Featherstone PR #4488.

Default mode verifies source excerpts transcribed from the existing downstream
review-fix branch 9f082bfe9c206e41f935fcba3dea29276b51ecc8. For an actual
checkout, pass --solver and --test-file. Neither mode imports Newton or Warp,
nor exercises CUDA/capture, device kernels, solver semantics, or performance.

Credit: CodeRabbit pointed out both review defects; the previous downstream
review patch implements them. This harness only checks a narrower source-level
preload/launch contract, and includes negative mutation controls.
"""
from __future__ import annotations
import argparse
import ast
from dataclasses import dataclass
import json
from pathlib import Path
import unittest
from types import SimpleNamespace

INSPECTED_ORIGIN = 'newton-physics/newton#4488@8f9c63f403def56d72fe8c0a7dcfe15a2fb58803'
INSPECTED_DOWNSTREAM = 'kvnloo/newton#25@9f082bfe9c206e41f935fcba3dea29276b51ecc8'

# Exact relevant control-flow and calls, omitting only unrelated statements.
# Fixture evidence is source-SHAPED; --solver validates the actual checkout.
FIXTURE_SOLVER = '''\
class SolverFeatherstone:
    def __init__(self, model):
        if self.use_tile_gemm:
            if self.fuse_cholesky and not self._has_joint_mimics:
                self.eval_inertia_matrix_cholesky_kernel = create_inertia_matrix_cholesky_kernel(
                    int(self.joint_count), int(self.dof_count)
                )
                tile_kernel = self.eval_inertia_matrix_cholesky_kernel
            else:
                self.eval_inertia_matrix_kernel = create_inertia_matrix_kernel(
                    int(self.joint_count), int(self.dof_count)
                )
                tile_kernel = self.eval_inertia_matrix_kernel
            wp.load_module(
                module=tile_kernel.module,
                device=model.device,
                block_dim=64 if self.fuse_cholesky and not self._has_joint_mimics else 256,
            )

    def step(self, model):
        if self.use_tile_gemm:
            if self.fuse_cholesky and not self._has_joint_mimics:
                wp.launch_tiled(
                    self.eval_inertia_matrix_cholesky_kernel,
                    dim=model.articulation_count,
                    inputs=[J_tiled, M_tiled, R_tiled],
                    outputs=[H_tiled, L_tiled],
                    device=model.device,
                    block_dim=64,
                )
            else:
                wp.launch_tiled(
                    self.eval_inertia_matrix_kernel,
                    dim=model.articulation_count,
                    inputs=[J_tiled, M_tiled],
                    outputs=[H_tiled],
                    device=model.device,
                    block_dim=256,
                )
'''
FIXTURE_TEST = '''\
import unittest
import warp as wp
import newton
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

def test_tile_gemm_loads_generated_module(test, device):
    model = _build_18_dof_chain(device)
    with wp.ScopedDevice("cpu"):
        for fuse_cholesky in (False, True):
            with test.subTest(fuse_cholesky=fuse_cholesky):
                solver = newton.solvers.SolverFeatherstone(
                    model, use_tile_gemm=True, fuse_cholesky=fuse_cholesky,
                )
                test.assertTrue(solver.use_tile_gemm)

devices = get_cuda_test_devices()
add_function_test(TestFeatherstoneTileGemm, "test_tile_gemm_loads_generated_module", test_tile_gemm_loads_generated_module, devices=devices)
'''


class SourceContractError(AssertionError):
    pass


def unparse(node):
    return ast.unparse(node)


def calls_named(node: ast.AST, name: str):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and unparse(n.func) == name]


def kw(call: ast.Call, name: str):
    return next((k.value for k in call.keywords if k.arg == name), None)


def unique(items, message):
    if len(items) != 1:
        raise SourceContractError(f'{message}: expected 1, found {len(items)}')
    return items[0]


def branch_with_two_launches(fn: ast.FunctionDef):
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and node.orelse:
            then_launches = [c for n in node.body for c in calls_named(n, 'wp.launch_tiled')]
            else_launches = [c for n in node.orelse for c in calls_named(n, 'wp.launch_tiled')]
            if len(then_launches) == 1 and len(else_launches) == 1:
                return node, then_launches[0], else_launches[0]
    raise SourceContractError('No mutually exclusive tile-launch pair found')


def attr_assigned_in_branch(branch: list[ast.stmt], variable: str):
    assignments = [n.value for stmt in branch for n in ast.walk(stmt)
                   if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == variable for t in n.targets)]
    return unique(assignments, f'assignment to {variable}')


def inspect_solver_source(source: str):
    tree = ast.parse(source)
    cls = unique([n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SolverFeatherstone'], 'SolverFeatherstone class')
    init = unique([n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__'], 'init method')
    step = unique([n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'step'], 'step method')
    load = unique(calls_named(init, 'wp.load_module'), 'tile preloader')
    if (mod := kw(load, 'module')) is None or unparse(mod) != 'tile_kernel.module':
        raise SourceContractError('Preload must target the generated tile kernel module')
    if (dev := kw(load, 'device')) is None or unparse(dev) != 'model.device':
        raise SourceContractError('Preload must explicitly use model.device')
    pre_dim = kw(load, 'block_dim')
    if pre_dim is None:
        raise SourceContractError('Preload is missing block_dim')
    launch_branch, launch_true, launch_false = branch_with_two_launches(step)
    if not isinstance(launch_branch.test, ast.BoolOp):
        raise SourceContractError('Unexpected launch guard')
    launch_guard = unparse(launch_branch.test)
    # Constructor selects the module under the same guard used by the step.
    constructor_guard = None
    for node in ast.walk(init):
        if isinstance(node, ast.If) and unparse(node.test) == launch_guard:
            then_kernel = attr_assigned_in_branch(node.body, 'tile_kernel')
            else_kernel = attr_assigned_in_branch(node.orelse, 'tile_kernel')
            constructor_guard = node
            break
    if constructor_guard is None:
        raise SourceContractError('Missing matching constructor kernel selection guard')
    expected = [(then_kernel, launch_true), (else_kernel, launch_false)]
    for assigned_kernel, launch in expected:
        if unparse(assigned_kernel) != unparse(launch.args[0]):
            raise SourceContractError(f'Preload chooses {unparse(assigned_kernel)}, but launch uses {unparse(launch.args[0])}')
        if kw(launch, 'device') is None or unparse(kw(launch, 'device')) != 'model.device':
            raise SourceContractError('Launch must explicitly use model.device')
    rows = []
    for fused in (False, True):
        for mimics in (False, True):
            obj = SimpleNamespace(fuse_cholesky=fused, _has_joint_mimics=mimics)
            env = {'self': obj}
            selected_branch = bool(eval(compile(ast.Expression(launch_branch.test), '<guard>', 'eval'), {'__builtins__': {}}, env))
            selected_launch = launch_true if selected_branch else launch_false
            selected_load = eval(compile(ast.Expression(pre_dim), '<preload>', 'eval'), {'__builtins__': {}}, env)
            selected_dim = eval(compile(ast.Expression(kw(selected_launch, 'block_dim')), '<launch>', 'eval'), {'__builtins__': {}}, env)
            if selected_load != selected_dim:
                raise SourceContractError(f'Variant mismatch for fused={fused} mimics={mimics}: preload {selected_load}, launch {selected_dim}')
            rows.append({'fuse_cholesky': fused, 'has_mimics': mimics, 'preload_block_dim': selected_load, 'launch_block_dim': selected_dim,
                         'kernel': unparse(selected_launch.args[0])})
    return {'rows': rows, 'launch_guard': launch_guard, 'preload_module': unparse(mod), 'device': unparse(dev)}


def inspect_cuda_test_source(source: str):
    tree = ast.parse(source)
    named_calls = [unparse(c.func) for c in ast.walk(tree) if isinstance(c, ast.Call)]
    if 'get_cuda_test_devices' not in named_calls:
        raise SourceContractError('Regression test is not explicitly CUDA-only')
    target = unique([n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'test_tile_gemm_loads_generated_module'], 'tile regression function')
    if any(isinstance(n, ast.If) and 'device.is_cuda' in unparse(n.test) for n in ast.walk(target)):
        raise SourceContractError('Regression still returns early on a CPU device')
    registrations = calls_named(tree, 'add_function_test')
    registration = unique(registrations, 'add_function_test registration')
    if kw(registration, 'devices') is None or unparse(kw(registration, 'devices')) != 'devices':
        raise SourceContractError('CUDA-filtered devices were not passed to test registration')
    return {'cuda_only_registration': True, 'requires_cuda_only': True, 'empty_cuda_skip_helper_source_inspected': True, 'tests_graph_capture': any(n.endswith('.ScopedCapture') for n in named_calls),
            'tests_step': any(n.endswith('.step') for n in named_calls)}


class ContractTests(unittest.TestCase):
    def test_fixture_four_preload_launch_variants(self):
        x = inspect_solver_source(FIXTURE_SOLVER)
        self.assertEqual([z['preload_block_dim'] for z in x['rows']], [256, 256, 64, 256])
        self.assertTrue(all(z['preload_block_dim'] == z['launch_block_dim'] for z in x['rows']))

    def test_fixture_cuda_skip_contract(self):
        x = inspect_cuda_test_source(FIXTURE_TEST)
        self.assertTrue(x['cuda_only_registration'])
        self.assertFalse(x['tests_graph_capture'])
        self.assertFalse(x['tests_step'])

    def test_negative_no_preload_dimension(self):
        x = FIXTURE_SOLVER.replace('block_dim=64 if self.fuse_cholesky and not self._has_joint_mimics else 256,', '')
        with self.assertRaisesRegex(SourceContractError, 'missing block_dim'):
            inspect_solver_source(x)

    def test_negative_wrong_preload_dimension(self):
        x = FIXTURE_SOLVER.replace('block_dim=64 if self.fuse_cholesky and not self._has_joint_mimics else 256,', 'block_dim=256,')
        with self.assertRaisesRegex(SourceContractError, 'Variant mismatch'):
            inspect_solver_source(x)

    def test_negative_wrong_preload_module(self):
        x = FIXTURE_SOLVER.replace('module=tile_kernel.module', 'module=wp.get_module()')
        with self.assertRaisesRegex(SourceContractError, 'generated tile kernel'):
            inspect_solver_source(x)

    def test_negative_wrong_preload_device(self):
        x = FIXTURE_SOLVER.replace('device=model.device,\n                block_dim', 'device=wp.get_device(),\n                block_dim')
        with self.assertRaisesRegex(SourceContractError, 'model.device'):
            inspect_solver_source(x)

    def test_negative_mismatched_kernel(self):
        x = FIXTURE_SOLVER.replace('self.eval_inertia_matrix_cholesky_kernel,\n                    dim=model.articulation_count', 'self.eval_inertia_matrix_kernel,\n                    dim=model.articulation_count')
        with self.assertRaisesRegex(SourceContractError, 'Preload chooses'):
            inspect_solver_source(x)

    def test_negative_branch_guard_drift(self):
        x = FIXTURE_SOLVER.replace('if self.fuse_cholesky and not self._has_joint_mimics:\n                wp.launch_tiled', 'if self.fuse_cholesky:\n                wp.launch_tiled')
        with self.assertRaisesRegex(SourceContractError, 'Unexpected launch guard'):
            inspect_solver_source(x)

    def test_negative_launch_dim_drift(self):
        x = FIXTURE_SOLVER.replace('device=model.device,\n                    block_dim=64,', 'device=model.device,\n                    block_dim=256,')
        with self.assertRaisesRegex(SourceContractError, 'Variant mismatch'):
            inspect_solver_source(x)

    def test_negative_cuda_false_pass(self):
        x = FIXTURE_TEST.replace('get_cuda_test_devices', 'get_test_devices').replace('model = _build_18_dof_chain(device)', 'if not device.is_cuda:\n        return\n    model = _build_18_dof_chain(device)')
        with self.assertRaisesRegex(SourceContractError, 'CUDA-only'):
            inspect_cuda_test_source(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver', type=Path, help='Real newton/_src/solvers/featherstone/solver_featherstone.py checkout')
    parser.add_argument('--test-file', type=Path, help='Real newton/tests/test_featherstone_tile_gemm.py checkout')
    parser.add_argument('--json', type=Path, help='Write verification report')
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    report = {'classification': 'source-only', 'origin': INSPECTED_ORIGIN, 'downstream': INSPECTED_DOWNSTREAM,
              'unittest_passed': result.testsRun if result.wasSuccessful() else 0, 'fixture': inspect_solver_source(FIXTURE_SOLVER),
              'test_registration': inspect_cuda_test_source(FIXTURE_TEST)}
    try:
        if args.solver:
            report['checkout_solver_contract'] = inspect_solver_source(args.solver.read_text())
        if args.test_file:
            report['checkout_test_contract'] = inspect_cuda_test_source(args.test_file.read_text())
    except (SourceContractError, SyntaxError) as exc:
        report['checkout_error'] = str(exc)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() and not report.get('checkout_error') else 1


if __name__ == '__main__':
    raise SystemExit(main())
