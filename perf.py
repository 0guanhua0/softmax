import time
from itertools import product
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from tinygrad import TinyJit
from tinygrad.device import Device
from tinygrad.helpers import GlobalCounters
from tinygrad.tensor import Tensor


def tune(
    sched: Callable,
    n: int,
    config: dict[str, list],
) -> dict:
    best_time, best_pair = float("inf"), None
    rng = np.random.default_rng()
    data = rng.standard_normal(size=(n), dtype=np.float32)

    key = list(config.keys())
    for x in product(*config.values()):
        pair = dict(zip(key, x))
        kwargs = {k: [v, 1, 1] for k, v in pair.items()}

        _, res = sched(n, data, **kwargs)
        del res
        Device[Device.DEFAULT].synchronize()

        st = time.perf_counter()
        _, res = sched(n, data, **kwargs)
        del res
        Device[Device.DEFAULT].synchronize()
        t = time.perf_counter() - st

        if t < best_time:
            best_time = t
            best_pair = pair

    return best_pair


def plot(res: dict):
    plt.style.use("dark_background")
    fig, ax = plt.subplots()
    for name, data in res.items():
        ax.plot(data["n"], data["gflops"], label=name)

    ax.set_xscale("log", base=2)
    ax.set_ylabel("gflops")
    ax.legend(loc="best")

    plt.savefig("perf.png")


def run(kernel):
    config = {
        "reduce_fat_ls": [2**i for i in range(4, 11)],
        "reduce_thin_ls": [2**i for i in range(4, 11)],
        "div_ls": [2**i for i in range(4, 11)],
    }
    n_values = [2**i for i in range(10, 20)]
    rng = np.random.default_rng()

    tuned_params_cache = {}
    if "fusion 5" in kernel and "fusion 5 tuned" in kernel:
        for n in n_values:
            tuned_params_cache[n] = tune(kernel["fusion 5"], n, config)

    res = {name: {"n": [], "gflops": []} for name in ["tinygrad"] + list(kernel.keys())}
    NUM_RUNS = 100

    for n in n_values:
        print(f"n={n} ({n / 1e6:.2f}M elements)")

        @TinyJit
        def tiny_jit(t: Tensor) -> Tensor:
            return t.softmax().realize()

        ops = -1
        timed_tensors = [
            Tensor(rng.standard_normal(size=(n), dtype=np.float32))
            for _ in range(NUM_RUNS)
        ]

        tiny_jit(timed_tensors[0]).realize()
        Device[Device.DEFAULT].synchronize()

        tiny_times = []
        for t in timed_tensors:
            GlobalCounters.reset()
            st = time.perf_counter()
            tiny_jit(t).realize()
            Device[Device.DEFAULT].synchronize()
            tiny_times.append(time.perf_counter() - st)
        tiny_time = np.median(tiny_times)
        ops = GlobalCounters.global_ops
        gflops = ops / tiny_time / 1e9
        res["tinygrad"]["n"].append(n)
        res["tinygrad"]["gflops"].append(gflops)
        print(f"{'tinygrad'} {gflops:.2f} GFLOPS (median time: {tiny_time:.4f}s)")

        for name, executor_func in kernel.items():
            executor = executor_func
            if name == "fusion 5 tuned":
                best_params = tuned_params_cache.get(n)
                best_params_kwargs = {k: [v, 1, 1] for k, v in best_params.items()}

                def _executor(n_inner, d1_inner, b_func=kernel["fusion 5"]):
                    return b_func(n_inner, d1_inner, **best_params_kwargs)

                executor = _executor

            d1_warmup = rng.standard_normal(size=(n), dtype=np.float32)
            _, out_warmup = executor(n, d1_warmup)
            expected = Tensor(d1_warmup).softmax().realize().numpy()
            np.testing.assert_allclose(out_warmup, expected, rtol=1e-5, atol=1e-5)
            Device[Device.DEFAULT].synchronize()
            del d1_warmup

            custom_times = []
            for _ in range(NUM_RUNS):
                d1_timed = rng.standard_normal(size=(n), dtype=np.float32)
                t, _ = executor(n, d1_timed)
                custom_times.append(t)
                del d1_timed
            median_time = np.median(custom_times)

            gflops = ops / median_time / 1e9
            res[name]["n"].append(n)
            res[name]["gflops"].append(gflops)
            params_str = ""
            if name == "fusion 5 tuned":
                params_str = f"(params: {tuned_params_cache.get(n)})"
            print(
                f"{name:<22}: {gflops:>7.2f} GFLOPS (median time: {median_time:.4f}s) {params_str}".rstrip()
            )

    plot(res)
