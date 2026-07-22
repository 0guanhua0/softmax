import gc
import itertools
import random
import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from tinygrad import TinyJit
from tinygrad.device import Device
from tinygrad.tensor import Tensor


def tune(func: Callable, n: int, arg: dict[str, list], run: int = 2**4) -> dict:
    best_time, best_kwargs = float("inf"), None
    rng = np.random.default_rng()
    data = rng.standard_normal(size=(n), dtype=np.float32)

    keys = list(arg.keys())
    for combo in itertools.product(*arg.values()):
        kwargs = dict(zip(keys, combo))

        ctx = func(n, data, **kwargs)

        _time = []
        for _ in range(run):
            Device[Device.DEFAULT].synchronize()
            t0 = time.perf_counter()
            ctx["run"]()
            Device[Device.DEFAULT].synchronize()
            _time.append(time.perf_counter() - t0)
        ctx["free"]()
        Device[Device.DEFAULT].allocator.free_cache()

        if np.median(_time) < best_time:
            best_time = np.median(_time)
            best_kwargs = kwargs

    return best_kwargs


def plot(res: dict):
    plt.style.use("dark_background")
    fig, ax = plt.subplots()
    for name, data in res.items():
        n = np.array(data["n"])
        g = np.array(data["gflops"])
        idx = np.argsort(n)
        n, g = n[idx], g[idx]

        ax.plot(n, g, label=name)

    ax.set_xscale("log", base=2)
    ax.set_ylabel("gflops")
    ax.legend(loc="best")

    plt.savefig("perf.png")


def run(kernel):
    arg = {
        "global_size": [[i, 1, 1] for i in range(8, 33)],
        "local_size": [[2**i, 1, 1] for i in range(8, 11)],
    }
    n_list = []
    for i in range(10, 25):
        n_list.append(2**i)
        n_list.extend(random.sample(range(2**i, 2 ** (i + 1)), 2**5))

    arg_tune = {}
    for k in list(kernel.keys()):
        arg_tune[k] = {}
        k_arg = {
            "global_size": arg["global_size"] if k == "7" else [[1, 1, 1]],
            "local_size": arg["local_size"],
        }
        for n in n_list:
            arg_tune[k][n] = tune(kernel[k], n, k_arg)

    res = {name: {"n": [], "gflops": []} for name in ["tinygrad"] + list(kernel.keys())}
    num_run = 2**6

    rng = np.random.default_rng()
    for n in n_list:
        print(f"n {n}")
        op = 5 * n
        data = rng.standard_normal(size=(n), dtype=np.float32)

        @TinyJit
        def tiny_jit(t: Tensor) -> Tensor:
            return t.softmax().realize()

        t_in = Tensor(data)
        _time = []
        for _ in range(num_run):
            Device[Device.DEFAULT].synchronize()
            t0 = time.perf_counter()
            tiny_jit(t_in).realize()
            Device[Device.DEFAULT].synchronize()
            _time.append(time.perf_counter() - t0)

        tiny_out = tiny_jit(t_in).realize().numpy()
        Device[Device.DEFAULT].synchronize()
        del t_in
        del tiny_jit

        gc.collect()
        Device[Device.DEFAULT].allocator.free_cache()
        gflops = op / np.median(_time) / 1e9
        res["tinygrad"]["n"].append(n)
        res["tinygrad"]["gflops"].append(gflops)
        print(f"tinygrad {gflops:.2f} gflops")

        for name, func in kernel.items():
            best_kwargs = arg_tune.get(name).get(n)
            ctx = func(n, data, **best_kwargs)

            _time = []
            for _ in range(num_run):
                Device[Device.DEFAULT].synchronize()
                t0 = time.perf_counter()
                ctx["run"]()
                Device[Device.DEFAULT].synchronize()
                _time.append(time.perf_counter() - t0)

            out = ctx["copyout"]()

            chunk_size = 2**20
            for i in range(0, n, chunk_size):
                np.testing.assert_allclose(
                    out[i : i + chunk_size],
                    tiny_out[i : i + chunk_size],
                    rtol=1e-5,
                    atol=1e-5,
                )

            ctx["free"]()
            Device[Device.DEFAULT].allocator.free_cache()

            gflops = op / np.median(_time) / 1e9
            res[name]["n"].append(n)
            res[name]["gflops"].append(gflops)
            print(f"{name} {gflops:.2f} gflops {best_kwargs}")

    plot(res)
