import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from tinygrad import TinyJit
from tinygrad.device import Device
from tinygrad.tensor import Tensor


def tune(
    func: Callable,
    n: int,
    arg: dict[str, list],
    run: int = 10,
) -> dict:
    best_time, best_pair = float("inf"), None
    rng = np.random.default_rng()
    data = rng.standard_normal(size=(n), dtype=np.float32)

    for k, val in arg.items():
        for v in val:
            kwargs = {"global_size": [1, 1, 1], k: v}

            _time = []
            for _ in range(run):
                st = time.perf_counter()
                func(n, data, **kwargs)
                Device[Device.DEFAULT].synchronize()
                _time.append(time.perf_counter() - st)

            t = np.median(_time)
            if t < best_time:
                best_time = t
                best_pair = {k: v}

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
    arg = {
        "local_size": [[2**i, 1, 1] for i in range(5, 11)],
    }
    n_list = [2**i for i in range(5, 28)]

    arg_tune = {}
    for k in list(kernel.keys()):
        arg_tune[k] = {}
        for n in n_list:
            arg_tune[k][n] = tune(kernel[k], n, arg)

    res = {name: {"n": [], "gflops": []} for name in ["tinygrad"] + list(kernel.keys())}
    num_run = 100

    rng = np.random.default_rng()
    for n in n_list:
        print(f"n {n}")
        op = 5 * n
        data = rng.standard_normal(size=(n), dtype=np.float32)

        @TinyJit
        def tiny_jit(t: Tensor) -> Tensor:
            return t.softmax().realize()

        _time = []
        for _ in range(num_run):
            st = time.perf_counter()
            tiny_out = tiny_jit(Tensor(data)).realize().numpy()
            Device[Device.DEFAULT].synchronize()
            _time.append(time.perf_counter() - st)
        gflops = op / np.median(_time) / 1e9
        res["tinygrad"]["n"].append(n)
        res["tinygrad"]["gflops"].append(gflops)
        print(f"tinygrad {gflops:.2f} gflops")

        for name, func in kernel.items():
            local_size = arg_tune.get(name).get(n).get("local_size")

            _time = []
            for _ in range(num_run):
                st = time.perf_counter()
                out = func(n, data, global_size=[1, 1, 1], local_size=local_size)
                Device[Device.DEFAULT].synchronize()
                _time.append(time.perf_counter() - st)
                np.testing.assert_allclose(out, tiny_out, rtol=1e-5, atol=1e-5)

            gflops = op / np.median(_time) / 1e9
            res[name]["n"].append(n)
            res[name]["gflops"].append(gflops)
            print(f"{name} {gflops:.2f} gflops")

    plot(res)
