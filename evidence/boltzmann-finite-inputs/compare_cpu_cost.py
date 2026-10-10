"""Bounded CPU-only cost comparison against the original public helper body."""

import json
import statistics
import time

import numpy as np
import warp as wp

import newton


@wp.func
def _original(a: float, b: float, alpha: float):
    e1 = wp.exp(alpha * a)
    e2 = wp.exp(alpha * b)
    return (a * e1 + b * e2) / (e1 + e2)


@wp.kernel
def _original_kernel(a: wp.array[float], b: wp.array[float], alpha: wp.array[float], result: wp.array[float]):
    i = wp.tid()
    result[i] = _original(a[i], b[i], alpha[i])


@wp.kernel
def _revised_kernel(a: wp.array[float], b: wp.array[float], alpha: wp.array[float], result: wp.array[float]):
    i = wp.tid()
    result[i] = newton.math.boltzmann(a[i], b[i], alpha[i])


def main():
    wp.init()
    count, trials, repeats = 65536, 16, 4
    rng = np.random.default_rng(20261010)
    cases = rng.uniform(-2.0, 2.0, (3, count)).astype(np.float32)
    inputs = [wp.array(values, dtype=float, device="cpu", requires_grad=True) for values in cases]
    seeds = wp.ones(count, dtype=float, device="cpu")
    states = {}
    for name, kernel in (("original", _original_kernel), ("revised", _revised_kernel)):
        result = wp.empty(count, dtype=float, device="cpu", requires_grad=True)
        with wp.Tape() as tape:
            wp.launch(kernel, dim=count, inputs=inputs, outputs=[result], device="cpu")
        tape.zero()
        tape.backward(grads={result: seeds})
        states[name] = (kernel, result, tape)
        states[name + "_values"] = result.numpy().copy()
        states[name + "_gradients"] = np.column_stack([tape.gradients[array].numpy().copy() for array in inputs])
        tape.zero()

    # Ordinary inputs remain within the original implementation's safe exponent range.
    np.testing.assert_allclose(states["revised_values"], states["original_values"], rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(states["revised_gradients"], states["original_gradients"], rtol=2e-5, atol=2e-6)
    # JIT, setup, and explicit warmup are outside the timed samples.
    for name in ("original", "revised"):
        kernel, result, tape = states[name]
        for _ in range(4):
            wp.launch(kernel, dim=count, inputs=inputs, outputs=[result], device="cpu")
            tape.backward(grads={result: seeds})
            tape.zero()
    wp.synchronize_device("cpu")
    samples = {mode: {name: [] for name in ("original", "revised")} for mode in ("forward", "backward_and_zero")}
    for trial in range(trials):
        order = ("original", "revised") if trial % 2 == 0 else ("revised", "original")
        for name in order:
            kernel, result, tape = states[name]
            start = time.perf_counter_ns()
            for _ in range(repeats):
                wp.launch(kernel, dim=count, inputs=inputs, outputs=[result], device="cpu")
            wp.synchronize_device("cpu")
            samples["forward"][name].append((time.perf_counter_ns() - start) / repeats / 1e6)
            start = time.perf_counter_ns()
            for _ in range(repeats):
                tape.backward(grads={result: seeds})
                tape.zero()
            wp.synchronize_device("cpu")
            samples["backward_and_zero"][name].append((time.perf_counter_ns() - start) / repeats / 1e6)

    summary = {
        "device": "cpu",
        "warp_version": wp.__version__,
        "count": count,
        "trials": trials,
        "repeats_per_trial": repeats,
        "warmup_launches_each": 4,
        "seed": 20261010,
        "input_range": [-2.0, 2.0],
        "values_and_gradients_match_original": True,
        "samples_ms_per_launch": samples,
        "median_ms_per_launch": {
            mode: {name: statistics.median(values) for name, values in timings.items()}
            for mode, timings in samples.items()
        },
        "notes": "CPU microbenchmark only; excludes JIT and setup, backward includes gradient reset; no CUDA or workload performance claim.",
    }
    for _mode, timings in summary["median_ms_per_launch"].items():
        timings["revised_over_original"] = timings["revised"] / timings["original"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
