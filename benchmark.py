import os
import random
import time
import unittest
from functools import cache
from itertools import product
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from tinygrad import TinyJit
from tinygrad.device import CompileError, Device
from tinygrad.helpers import GlobalCounters, flat_mv
from tinygrad.runtime.ops_metal import MetalAllocator, MetalProgram
from tinygrad.tensor import Tensor


@cache
def compile(src: str) -> bytes:
    if os.environ.get("PRINT_KERNEL"):
        print(src)
    return Device[Device.DEFAULT].compiler.compile(src)


metalalloc = MetalAllocator(Device[Device.DEFAULT])

base = """
#include <metal_stdlib>
using namespace metal;

kernel void base(device float *data_out, device float *data_in, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    size_t gid0 = gid.x;

    float max_val = data_in[0];
    for (size_t i = 1; i < {global_size[0]}; ++i) {{
        max_val = fmax(max_val, data_in[i]);
    }}

    float sum_exp = 0.0f;
    for (size_t i = 0; i < {global_size[0]}; ++i) {{
        sum_exp += exp(data_in[i] - max_val);
    }}

    data_out[gid0] = exp(data_in[gid0] - max_val) / sum_exp;
}}
"""

fast_exp = """
#include <metal_stdlib>
using namespace metal;

inline float fast_exp(float y) {{
    float x = y * 1.44269504f;
    float x_int_f = round(x);
    float x_frac = x - x_int_f;
    float p = fma(x_frac, fma(x_frac, fma(x_frac, 0.05700169f, 0.24858144f), 0.69282515f), 0.99916080f);
    int x_int = (int)x_int_f;
    int biased_exp = x_int + 127;
    if (biased_exp <= 0) return 0.0f;
    return as_type<float>(biased_exp << 23) * p;
}}
"""

generic_reduce_max = """
#include <metal_stdlib>
using namespace metal;

kernel void generic_reduce_max(device float *data_out, device const float *data_in, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    uint gid0 = gid.x;
    uint lid0 = lid.x;

    float p_max = -FLT_MAX;
    size_t chunk_start = (size_t)gid0 * {items_per_workgroup};
    size_t chunk_end = chunk_start + {items_per_workgroup};

    for (size_t i = chunk_start + lid0; i < chunk_end; i += {local_size[0]}) {{
        if (i < {n}) {{ // Guard against reading out of bounds.
            p_max = fmax(p_max, data_in[i]);
        }}
    }}

    // 2. Perform a parallel reduction within the workgroup on the partial maximums.
    threadgroup float local_data[{local_size[0]}];
    local_data[lid0] = p_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = {local_size[0]} / 2; stride > 0; stride /= 2) {{
        if (lid0 < stride) {{
            local_data[lid0] = fmax(local_data[lid0], local_data[lid0 + stride]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // 3. Thread 0 writes the workgroup's final result.
    if (lid0 == 0) {{
        data_out[gid0] = local_data[0];
    }}
}}
"""

generic_reduce_sum = """
#include <metal_stdlib>
using namespace metal;

kernel void generic_reduce_sum(device float *data_out, device const float *data_in, device const float* global_max_val, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    uint gid0 = gid.x;
    uint lid0 = lid.x;
    float max_val = *global_max_val;

    // 1. Each thread computes a partial sum over its assigned chunk of data.
    float p_sum = 0.0f;
    size_t chunk_start = (size_t)gid0 * {items_per_workgroup};
    size_t chunk_end = chunk_start + {items_per_workgroup};

    for (size_t i = chunk_start + lid0; i < chunk_end; i += {local_size[0]}) {{
        if (i < {n}) {{ // Guard against reading out of bounds.
            p_sum += exp(data_in[i] - max_val);
        }}
    }}

    // 2. Perform a parallel reduction within the workgroup on the partial sums.
    threadgroup float local_data[{local_size[0]}];
    local_data[lid0] = p_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = {local_size[0]} / 2; stride > 0; stride /= 2) {{
        if (lid0 < stride) {{
            local_data[lid0] += local_data[lid0 + stride];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // 3. Thread 0 writes the workgroup's final result.
    if (lid0 == 0) {{
        data_out[gid0] = local_data[0];
    }}
}}
"""

final_division = """
#include <metal_stdlib>
using namespace metal;

kernel void final_division(device float *data_out, device const float *data_in, device const float *reduction_results, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    float max_val = *reduction_results;
    float sum_exp = *(reduction_results + 1);
    float inv_sum_exp = 1.0f / sum_exp;

    for (size_t i = (size_t)gid.x * {local_size[0]} + lid.x; i < {n}; i += (size_t){global_size[0]} * {local_size[0]}) {{
        data_out[i] = exp(data_in[i] - max_val) * inv_sum_exp;
    }}
}}
"""


"""
kernel fusion 5

- local_max
- global_max
- local_sum
- global_sum
- global_div
"""

fusion5_reg_reduce_max_kernel = """
#include <metal_stdlib>
using namespace metal;
kernel void fusion5_reg_reduce_max(device float *data_out, device const float *data_in, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    float thread_max = -FLT_MAX;
    size_t block_start_idx = (size_t)gid.x * {items_per_block};
    size_t block_end_idx = block_start_idx + {items_per_block};

    // Each thread reduces a strided chunk of floats from global memory. No bounds check needed due to input padding.
    for (size_t i = block_start_idx + lid.x; i < block_end_idx; i += {local_size[0]}) {{
        thread_max = fmax(thread_max, data_in[i]);
    }}

    // Workgroup-level reduction on `thread_max` using SIMD shuffles
    threadgroup float scratch_pad[{local_size[0]} / {threads_per_simdgroup}];
    uint simd_group_id = lid.x / {threads_per_simdgroup};
    uint simd_lane_id = lid.x % {threads_per_simdgroup};
    uint num_simd_groups = {local_size[0]} / {threads_per_simdgroup};

    for (uint s = {threads_per_simdgroup} / 2; s > 0; s /= 2) {{
        thread_max = fmax(thread_max, simd_shuffle_xor(thread_max, s));
    }}

    if (simd_lane_id == 0) scratch_pad[simd_group_id] = thread_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {{
        thread_max = (simd_lane_id < num_simd_groups) ? scratch_pad[simd_lane_id] : -FLT_MAX;
        for (uint s = {threads_per_simdgroup} / 2; s > 0; s /= 2) {{
            thread_max = fmax(thread_max, simd_shuffle_xor(thread_max, s));
        }}
    }}

    if (lid.x == 0) {{
        data_out[gid.x] = thread_max;
    }}
}}
"""

fusion5_reg_reduce_sum_kernel = """
#include <metal_stdlib>
using namespace metal;
kernel void fusion5_reg_reduce_sum(device float *data_out, device const float *data_in, device const float* global_max_val_buf, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    float global_max = *global_max_val_buf;
    float thread_sum = 0.0f;
    size_t block_start_idx = (size_t)gid.x * {items_per_block};
    size_t block_end_idx = block_start_idx + {items_per_block};

    // Each thread computes a partial sum over its assigned chunk of data. No bounds check needed due to padding.
    for (size_t i = block_start_idx + lid.x; i < block_end_idx; i += {local_size[0]}) {{
        thread_sum += exp(data_in[i] - global_max);
    }}

    // Workgroup-level reduction for sum using SIMD shuffles
    threadgroup float scratch_pad[{local_size[0]} / {threads_per_simdgroup}];
    uint simd_group_id = lid.x / {threads_per_simdgroup};
    uint simd_lane_id = lid.x % {threads_per_simdgroup};
    uint num_simd_groups = {local_size[0]} / {threads_per_simdgroup};

    for (uint s = {threads_per_simdgroup} / 2; s > 0; s /= 2) {{
        thread_sum += simd_shuffle_xor(thread_sum, s);
    }}
    if (simd_lane_id == 0) scratch_pad[simd_group_id] = thread_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {{
        thread_sum = (simd_lane_id < num_simd_groups) ? scratch_pad[simd_lane_id] : 0.0f;
        for (uint s = {threads_per_simdgroup} / 2; s > 0; s /= 2) {{
            thread_sum += simd_shuffle_xor(thread_sum, s);
        }}
    }}

    if (lid.x == 0) {{
        data_out[gid.x] = thread_sum;
    }}
}}
"""

fusion5_vec_reduce_max_kernel = """
#include <metal_stdlib>
using namespace metal;
kernel void fusion5_vec_reduce_max(device float *data_out, device const float *data_in, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    float thread_max = -FLT_MAX;
    size_t block_start_vec_idx = (size_t)gid.x * ({items_per_block} / 4);
    device const float4* data_in_vec = (device const float4*)data_in;

    // Each thread reduces a strided chunk of float4s from global memory. No bounds check needed due to input padding.
    for (uint i = lid.x; i < ({items_per_block} / 4); i += {local_size[0]}) {{
        float4 val = data_in_vec[block_start_vec_idx + i];
        thread_max = fmax(thread_max, fmax(fmax(val.x, val.y), fmax(val.z, val.w)));
    }}

    // Workgroup-level reduction on `thread_max` using traditional shared memory approach
    threadgroup float local_data[{local_size[0]}];
    local_data[lid.x] = thread_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = {local_size[0]} / 2; stride > 0; stride /= 2) {{
        if (lid.x < stride) {{
            local_data[lid.x] = fmax(local_data[lid.x], local_data[lid.x + stride]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (lid.x == 0) {{
        data_out[gid.x] = local_data[0];
    }}
}}
"""

fusion5_vec_reduce_sum_kernel = """
#include <metal_stdlib>
using namespace metal;
kernel void fusion5_vec_reduce_sum(device float *data_out, device const float *data_in, device const float* global_max_val_buf, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    float global_max = *global_max_val_buf;
    float thread_sum = 0.0f;

    // Calculate start index for this block
    size_t block_start_vec_idx = (size_t)gid.x * ({items_per_block} / 4);
    device const float4* data_in_vec = (device const float4*)data_in;
    float4 global_max_vec = float4(global_max);

    // 1. Each thread reduces a strided chunk of float4s directly from global memory.
    // This avoids the large shared memory buffer, improving occupancy.
    for (uint i = lid.x; i < ({items_per_block} / 4); i += {local_size[0]}) {{
        float4 val = data_in_vec[block_start_vec_idx + i] - global_max_vec;
        float4 exp_val;
        exp_val.x = exp(val.x);
        exp_val.y = exp(val.y);
        exp_val.z = exp(val.z);
        exp_val.w = exp(val.w);
        thread_sum += (exp_val.x + exp_val.y + exp_val.z + exp_val.w);
    }}

    // 2. Workgroup-level reduction for sum using traditional shared memory.
    // This part is fine as it uses a small, fixed-size buffer.
    threadgroup float local_data[{local_size[0]}];
    local_data[lid.x] = thread_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = {local_size[0]} / 2; stride > 0; stride /= 2) {{
        if (lid.x < stride) {{
            local_data[lid.x] += local_data[lid.x + stride];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // 3. Thread 0 writes the final result for the block.
    if (lid.x == 0) {{
        data_out[gid.x] = local_data[0];
    }}
}}
"""

global_reduce_max_vector_kernel = """
#include <metal_stdlib>
using namespace metal;

kernel void global_reduce_max_vector(device float *data_out, device const float *data_in, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    threadgroup float local_data[{local_size[0]}];
    float p_max = -FLT_MAX;
    size_t global_thread_idx = (size_t)gid.x * {local_size[0]} + lid.x;
    size_t start_idx = global_thread_idx * 4;

    // Vectorized load with tail handling. Each thread is responsible for up to 4 elements.
    if (start_idx < {n}) {{
        device const float4* data_in_vec = (device const float4*)(data_in + start_idx);
        float4 val = *data_in_vec;

        if ({n} - start_idx == 1) {{
            p_max = val.x;
        }} else if ({n} - start_idx == 2) {{
            p_max = fmax(val.x, val.y);
        }} else if ({n} - start_idx == 3) {{
            p_max = fmax(fmax(val.x, val.y), val.z);
        }} else {{
            p_max = fmax(fmax(val.x, val.y), fmax(val.z, val.w));
        }}
    }}

    local_data[lid.x] = p_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = {local_size[0]} / 2; stride > 0; stride /= 2) {{
        if (lid.x < stride) {{
            local_data[lid.x] = fmax(local_data[lid.x], local_data[lid.x + stride]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (lid.x == 0) {{
        data_out[gid.x] = local_data[0];
    }}
}}
"""

global_reduce_sum_vector_kernel = """
#include <metal_stdlib>
using namespace metal;

kernel void global_reduce_sum_vector(device float *data_out, device const float *data_in, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    threadgroup float local_data[{local_size[0]}];
    float p_sum = 0.0f;
    size_t global_thread_idx = (size_t)gid.x * {local_size[0]} + lid.x;
    size_t start_idx = global_thread_idx * 4;

    if (start_idx < {n}) {{
        device const float4* data_in_vec = (device const float4*)(data_in + start_idx);
        float4 val = *data_in_vec;

        p_sum += val.x;
        if (start_idx + 1 < {n}) p_sum += val.y;
        if (start_idx + 2 < {n}) p_sum += val.z;
        if (start_idx + 3 < {n}) p_sum += val.w;
    }}

    local_data[lid.x] = p_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = {local_size[0]} / 2; stride > 0; stride /= 2) {{
        if (lid.x < stride) {{
            local_data[lid.x] += local_data[lid.x + stride];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (lid.x == 0) {{
        data_out[gid.x] = local_data[0];
    }}
}}
"""


copy_sum_kernel = """
#include <metal_stdlib>
using namespace metal;
kernel void copy_sum(device float *data_out, device const float *data_in) {
  data_out[1] = data_in[0];
}
"""

local_max_and_sum = """
#include <metal_stdlib>
using namespace metal;

kernel void local_max_and_sum(device float *data_out, device float *data_in, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    uint gid0 = gid.x; /* global_size[0] == num_workgroup */
    uint lid0 = lid.x; /* local_size[0] */
    size_t global_idx = (size_t
)gid0 * {local_size[0]} + lid0;

    threadgroup float local_values[{local_size[0]}];
    threadgroup float temp_storage[{local_size[0]}];

    // 1. Load data into threadgroup memory once.
    local_values[lid0] = data_in[global_idx];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Find local max without destroying original values.
    temp_storage[lid0] = local_values[lid0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = {local_size[0]} / 2; stride > 0; stride /= 2) {{
        if (lid0 < stride) {{
            temp_storage[lid0] = fmax(temp_storage[lid0], temp_storage[lid0 + stride]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float local_max = temp_storage[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Compute exp(x - local_max) using cached values and perform sum reduction.
    local_values[lid0] = exp(local_values[lid0] - local_max);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = {local_size[0]} / 2; stride > 0; stride /= 2) {{
        if (lid0 < stride) {{
            local_values[lid0] += local_values[lid0 + stride];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float local_sum = local_values[0];

    // 4. Write out results with coalesced access.
    if (lid0 == 0) {{
        data_out[gid0] = local_max;
        data_out[gid0 + {num_workgroup}] = local_sum;
    }}
}}
"""

global_reduce_max = """
#include <metal_stdlib>
using namespace metal;

kernel void global_reduce_max(device float *data_out, device const float *data_in, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    uint gid0 = gid.x;
    uint lid0 = lid.x;
    size_t global_idx = (size_t)gid0 * {local_size[0]} + lid0;

    threadgroup float local_data[{local_size[0]}];
    local_data[lid0] = (global_idx < {n}) ? data_in[global_idx] : -FLT_MAX;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = {local_size[0]} / 2; stride > 0; stride /= 2) {{
        if (lid0 < stride) {{
            local_data[lid0] = fmax(local_data[lid0], local_data[lid0 + stride]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (lid0 == 0) {{
        data_out[gid0] = local_data[0];
    }}
}}
"""
global_sum = """
#include <metal_stdlib>
using namespace metal;

kernel void global_sum(device float *data_out, device const float *data_in, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    uint gid0 = gid.x;
    uint lid0 = lid.x;
    size_t global_idx = (size_t)gid0 * {local_size[0]} + lid0;

    threadgroup float local_data[{local_size[0]}];
    local_data[lid0] = (global_idx < {n}) ? data_in[global_idx] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = {local_size[0]} / 2; stride > 0; stride /= 2) {{
        if (lid0 < stride) {{
            local_data[lid0] += local_data[lid0 + stride];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (lid0 == 0) {{
        data_out[gid0] = local_data[0];
    }}
}}
"""
global_div = """
#include <metal_stdlib>
using namespace metal;

kernel void global_div(device float *data_out, device float *data_in0, device const float *reduction_results, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    uint gid0 = gid.x; /* global_size[0] */
    uint lid0 = lid.x; /* local_size[0] */

    size_t global_idx = (size_t)gid0 * {local_size[0]} + lid0;

    float max_val = *reduction_results;
    float sum_exp = *(reduction_results + 1);
    *(data_out + global_idx) = exp(*(data_in0 + global_idx) - max_val) / sum_exp;
}}
"""
adjust_sums_kernel = """
#include <metal_stdlib>
using namespace metal;

kernel void adjust_sums(device float *adjusted_sums_out, device const float *intermediate_buf, device const float* global_max_val_buf, uint3 gid [[threadgroup_position_in_grid]]) {{
    // intermediate_buf layout: [max_0, ..., max_{{N-1}}, sum_0, ..., sum_{{N-1}}]
    // N = num_workgroup
    uint num_wg = {num_workgroup};
    float local_max = intermediate_buf[gid.x];
    float local_sum = intermediate_buf[gid.x + num_wg];

    adjusted_sums_out[gid.x] = local_sum * exp(local_max - global_max_val_buf[0]);
}}
"""

global_div_opt = """
#include <metal_stdlib>
using namespace metal;
kernel void global_div_opt(device float *data_out, device const float *data_in0, device const float *reduction_results, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    float max_val = *reduction_results;
    float sum_exp = *(reduction_results + 1);
    float inv_sum_exp = 1.0f / sum_exp;

    // Each thread processes a chunk of ITEMS_PER_THREAD elements
    size_t chunk_idx = (size_t)gid.x * {local_size[0]} + lid.x;
    size_t start_elem_idx = chunk_idx * {ITEMS_PER_THREAD};

    device const float4* data_in_vec = (device const float4*)(data_in0 + start_elem_idx);
    device float4* data_out_vec = (device float4*)(data_out + start_elem_idx);
    float4 max_val_vec = float4(max_val);

    // Loop to process the chunk, vectorized by 4
    for (uint i = 0; i < ({ITEMS_PER_THREAD} / 4); ++i) {{
        float4 val = data_in_vec[i];
        float4 diff = val - max_val_vec;
        float4 exp_diff;
        exp_diff.x = exp(diff.x);
        exp_diff.y = exp(diff.y);
        exp_diff.z = exp(diff.z);
        exp_diff.w = exp(diff.w);
        data_out_vec[i] = exp_diff * inv_sum_exp;
    }}
}}
"""

final_division_fast_exp = fast_exp + final_division.replace("exp(", "fast_exp(")
generic_reduce_sum_fast_exp = fast_exp + generic_reduce_sum.replace("exp(", "fast_exp(")


def fine_tune_scheduler(
    scheduler_func: Callable,
    n: int,
    param_configs: dict[str, list],
    verbose: bool = True,
) -> tuple[float, dict]:
    if verbose:
        print(f"\n--- Fine-tuning '{scheduler_func.__name__}' for n={n} via Grid Search over params: {list(param_configs.keys())} ---")

    best_time = float('inf')
    best_params_combo = None
    rng = np.random.default_rng()
    d1_timed = rng.standard_normal(size=(n), dtype=np.float32)

    param_names = list(param_configs.keys())
    param_values_list = list(param_configs.values())
    all_combinations = list(product(*param_values_list))

    if verbose:
        print(f"  Testing {len(all_combinations)} combinations...")

    for i, combo in enumerate(all_combinations):
        current_params = dict(zip(param_names, combo))
        kwargs = {k: [v, 1, 1] for k, v in current_params.items()}
        try:
            # Warm-up run for kernel compilation. Result is discarded.
            _, res = scheduler_func(n, d1_timed, **kwargs)
            del res
            Device[Device.DEFAULT].synchronize()

            # Timed run. Result is discarded.
            st = time.perf_counter()
            _, res = scheduler_func(n, d1_timed, **kwargs)
            del res
            Device[Device.DEFAULT].synchronize()
            run_time = time.perf_counter() - st

            if verbose:
                print(f"    Trying combo {i+1}/{len(all_combinations)}: {current_params} -> {run_time:.4f}s")

            if run_time < best_time:
                best_time = run_time
                best_params_combo = current_params
        except Exception as e:
            if verbose:
                print(f"    Trying {current_params}: FAILED ({type(e).__name__})")

    if best_params_combo is None:
        raise RuntimeError(f"Tuning failed for {scheduler_func.__name__}. No valid parameter combination found.")

    if verbose:
        print(f"--- Best overall parameters found for n={n}: {best_params_combo} ({best_time:.4f}s) ---")

    return best_time, best_params_combo

@unittest.skipIf(Device.DEFAULT != "METAL", "metal kernel")
class Softmax(unittest.TestCase):
    def _driver(self, name: str, n: int, custom_executor: Callable[[int, np.ndarray], tuple[float, np.ndarray]], rtol=1e-6, atol=1e-6):
        NUM_RUNS = 2 ** 0
        rng = np.random.default_rng()
        d1_warmup = rng.standard_normal(size=(n), dtype=np.float32)
        d1_for_verification = rng.standard_normal(size=(n), dtype=np.float32)

        # 2. Run the custom implementation FIRST
        # WARM UP
        custom_executor(n, d1_warmup)
        Device[Device.DEFAULT].synchronize() # Explicitly wait for GPU to finish warm-up run.

        # TIMED RUN
        custom_times = []
        for _ in range(NUM_RUNS):
            # Regenerate data for each run to flush cache
            d1_timed_custom = rng.standard_normal(size=(n), dtype=np.float32)
            run_time, _ = custom_executor(n, d1_timed_custom)
            custom_times.append(run_time)
        custom_time = np.median(custom_times)

        # Get the custom result for verification. The custom_executor frees its own resources.
        _, d0 = custom_executor(n, d1_for_verification)

        # 3. Run tinygrad baseline to get correct result and timing
        @TinyJit
        def tiny_jit(t: Tensor) -> Tensor:
            return t.softmax().realize()

        try:
            # WARM UP
            tiny_jit(Tensor(d1_warmup)).realize()
            Device[Device.DEFAULT].synchronize()

            # TIMED RUN for tinygrad
            tiny_times = []
            for _ in range(NUM_RUNS):
                d1_timed = rng.standard_normal(size=(n), dtype=np.float32)
                GlobalCounters.reset()
                st_tiny = time.perf_counter()
                tiny_jit(Tensor(d1_timed)).realize()
                Device[Device.DEFAULT].synchronize()
                tiny_times.append(time.perf_counter() - st_tiny)

            tiny_time = np.median(tiny_times)
            # The op count is consistent for a given shape, so getting it from the last run is fine.
            ops = GlobalCounters.global_ops

            # Get the tinygrad result for verification using the same dedicated input
            tiny_softmax_tensor = tiny_jit(Tensor(d1_for_verification))
            Device[Device.DEFAULT].synchronize()
            tiny_softmax = tiny_softmax_tensor.numpy()
        except (CompileError, RuntimeError) as e: # Catch runtime errors for OOM etc.
            print(f"Tinygrad failed for N={n}: {e}")
            tiny_time, ops, tiny_softmax = float('nan'), -1, None

        # 4. Verify results
        if ops != -1 and tiny_softmax is not None:
            np.testing.assert_allclose(d0, tiny_softmax, rtol=rtol, atol=atol)


        # 5. Print results
        title_suffix = f" ({name})" if name else ""
        print(
            f"\nSoftmax {n}{title_suffix}\n"
            + f"custom median time ({NUM_RUNS} runs): {custom_time:.4f}s, GFLOPS: {(ops / custom_time / 1e9) if ops != -1 and custom_time > 0 else 0.00:.2f}\n"
        )
        if ops != -1:
          print(f"tinygrad median time ({NUM_RUNS} runs): {tiny_time:.4f}s, GFLOPS: {ops / tiny_time / 1e9:.2f}\n")

    def _sched_base(self, n: int, d1: np.ndarray) -> tuple[float, np.ndarray]:
        if n == 0: return 0.0, np.array([], dtype=np.float32)
        global_size = [n, 1, 1]
        prog = MetalProgram(Device[Device.DEFAULT], "base", compile(base.format(global_size=global_size)))

        data_out_size = n * 4
        data_in_size = n * 4
        data_out = metalalloc.alloc(data_out_size)
        data_in = metalalloc.alloc(data_in_size)
        metalalloc._copyin(data_in, d1.tobytes())

        st_custom = time.perf_counter()
        prog(data_out, data_in, global_size=global_size, local_size=[1, 1, 1], wait=True)
        custom_time = time.perf_counter() - st_custom

        d0 = np.empty((n), dtype=np.float32)
        metalalloc._copyout(flat_mv(d0.data), data_out)

        metalalloc.free(data_out, data_out_size)
        metalalloc.free(data_in, data_in_size)

        return custom_time, d0
    def _sched_fusion_5(self, n: int, d1: np.ndarray, **kwargs) -> tuple[float, np.ndarray]:
        if n == 0: return 0.0, np.array([], dtype=np.float32)

        default_reduce_ls = [256, 1, 1]
        default_div_ls = [256, 1, 1]

        reduce_fat_ls = kwargs.get("reduce_fat_ls", default_reduce_ls)
        reduce_thin_ls = kwargs.get("reduce_thin_ls", default_reduce_ls)
        div_ls = kwargs.get("div_ls", default_div_ls)

        # 1. Setup buffers and parameters
        max_workgroups = 16384 # A safe cap for grid size

        # Buffers
        data_in_buf_size = n * 4
        data_out_buf_size = n * 4
        temp_buf_size = max_workgroups * 4
        reduction_results_buf_size = 2 * 4

        data_in_buf = metalalloc.alloc(data_in_buf_size)
        metalalloc._copyin(data_in_buf, d1.tobytes())
        data_out_buf = metalalloc.alloc(data_out_buf_size)
        temp_buf_A = metalalloc.alloc(temp_buf_size)
        temp_buf_B = metalalloc.alloc(temp_buf_size)
        reduction_results_buf = metalalloc.alloc(reduction_results_buf_size)

        prog_copy_sum = MetalProgram(Device[Device.DEFAULT], "copy_sum", compile(copy_sum_kernel))
        st_custom = time.perf_counter()

        # Stage 1 & 2: Find Global Max
        if n > 0:
            local_size_max1 = reduce_fat_ls
            items_per_thread_max1 = (n + max_workgroups * local_size_max1[0] - 1) // (max_workgroups * local_size_max1[0]) or 1
            items_per_workgroup_max1 = items_per_thread_max1 * local_size_max1[0]
            num_wg_stage1 = (n + items_per_workgroup_max1 - 1) // items_per_workgroup_max1
            if num_wg_stage1 > max_workgroups: num_wg_stage1 = max_workgroups

            params_max_stage1 = {"n": f"{n}UL", "local_size": local_size_max1, "items_per_workgroup": items_per_workgroup_max1}
            prog_max_stage1 = MetalProgram(Device[Device.DEFAULT], "generic_reduce_max", compile(generic_reduce_max.format(**params_max_stage1)))
            prog_max_stage1(temp_buf_A, data_in_buf, global_size=[num_wg_stage1, 1, 1], local_size=local_size_max1)

            # Stage 2: Recursive reduction on partial maxes in temp_buf_A
            src_buf_max, current_size_max = temp_buf_A, num_wg_stage1
            i_max = 0
            temp_bufs_max = [temp_buf_B, temp_buf_A]
            while current_size_max > 1:
                local_size_thin = reduce_thin_ls
                num_wg_thin = (current_size_max + local_size_thin[0] - 1) // local_size_thin[0]
                params_max_thin = {"n": f"{current_size_max}UL", "local_size": local_size_thin}
                prog_max_thin = MetalProgram(Device[Device.DEFAULT], "global_reduce_max", compile(global_reduce_max.format(**params_max_thin)))
                dest_buf_max = temp_bufs_max[i_max % 2]
                prog_max_thin(dest_buf_max, src_buf_max, global_size=[num_wg_thin, 1, 1], local_size=local_size_thin)
                src_buf_max = dest_buf_max
                current_size_max = num_wg_thin
                i_max += 1
            final_max_buf = src_buf_max
            metalalloc._transfer(reduction_results_buf, final_max_buf, 4, src_dev=metalalloc.dev, dest_dev=metalalloc.dev)

        # Stage 3 & 4: Find Global Sum
        num_wg_stage3 = 0
        if n > 0:
            # Stage 3: First-level reduction for sum (data -> partial sums)
            local_size_sum1 = reduce_fat_ls
            items_per_thread_sum1 = (n + max_workgroups * local_size_sum1[0] - 1) // (max_workgroups * local_size_sum1[0]) or 1
            items_per_workgroup_sum1 = items_per_thread_sum1 * local_size_sum1[0]
            num_wg_stage3 = (n + items_per_workgroup_sum1 - 1) // items_per_workgroup_sum1
            if num_wg_stage3 > max_workgroups: num_wg_stage3 = max_workgroups
            params_sum_stage3 = {"n": f"{n}UL", "local_size": local_size_sum1, "items_per_workgroup": items_per_workgroup_sum1}
            prog_sum_stage3 = MetalProgram(Device[Device.DEFAULT], "generic_reduce_sum", compile(generic_reduce_sum.format(**params_sum_stage3)))
            prog_sum_stage3(temp_buf_A, data_in_buf, reduction_results_buf, global_size=[num_wg_stage3, 1, 1], local_size=local_size_sum1)

        # Stage 4: Recursive reduction on partial sums in temp_buf_A
        if num_wg_stage3 > 0:
            src_buf_sum, current_size_sum = temp_buf_A, num_wg_stage3
            i_sum = 0
            temp_bufs_sum = [temp_buf_B, temp_buf_A]
            while current_size_sum > 1:
                local_size_thin = reduce_thin_ls
                num_wg_thin = (current_size_sum + local_size_thin[0] - 1) // local_size_thin[0]
                params_sum_thin = {"n": f"{current_size_sum}UL", "local_size": local_size_thin}
                prog_sum_thin = MetalProgram(Device[Device.DEFAULT], "global_sum", compile(global_sum.format(**params_sum_thin)))
                dest_buf_sum = temp_bufs_sum[i_sum % 2]
                prog_sum_thin(dest_buf_sum, src_buf_sum, global_size=[num_wg_thin, 1, 1], local_size=local_size_thin)
                src_buf_sum = dest_buf_sum
                current_size_sum = num_wg_thin
                i_sum += 1
            # After reduction, src_buf_sum holds the buffer with the final sum. Copy it.
            prog_copy_sum(reduction_results_buf, src_buf_sum, global_size=[1,1,1], local_size=[1,1,1])

        # Stage 5: Final division
        if n > 0:
            final_div_gs = min(max_workgroups, (n + div_ls[0] - 1) // div_ls[0])
            prog_final_div = MetalProgram(Device[Device.DEFAULT], "final_division", compile(final_division.format(local_size=div_ls, global_size=[final_div_gs,1,1], n=f"{n}UL")))
            prog_final_div(data_out_buf, data_in_buf, reduction_results_buf, global_size=[final_div_gs, 1, 1], local_size=div_ls)

        Device[Device.DEFAULT].synchronize()
        custom_time = time.perf_counter() - st_custom

        d0 = np.empty((n), dtype=np.float32)
        if n > 0:
            metalalloc._copyout(flat_mv(d0.data), data_out_buf)

        metalalloc.free(data_in_buf, data_in_buf_size)
        metalalloc.free(data_out_buf, data_out_buf_size)
        metalalloc.free(temp_buf_A, temp_buf_size)
        metalalloc.free(temp_buf_B, temp_buf_size)
        metalalloc.free(reduction_results_buf, reduction_results_buf_size)

        return custom_time, d0
    def _sched_fusion_5_fast_exp(self, n: int, d1: np.ndarray) -> tuple[float, np.ndarray]:
        if n == 0: return 0.0, np.array([], dtype=np.float32)

        # Re-implementing based on the correct multi-stage reduction logic from _sched_fusion_5,
        # but using fast_exp kernels. The previous implementation had a logic error in its
        # generic `reduce_op` helper.
        reduce_fat_ls = [256, 1, 1]
        reduce_thin_ls = [256, 1, 1]
        div_ls = [256, 1, 1]

        # 1. Setup buffers and parameters
        max_workgroups = 16384 # A safe cap for grid size

        # Buffers
        data_in_buf_size = n * 4
        data_out_buf_size = n * 4
        temp_buf_size = max_workgroups * 4
        reduction_results_buf_size = 2 * 4

        data_in_buf = metalalloc.alloc(data_in_buf_size)
        metalalloc._copyin(data_in_buf, d1.tobytes())
        data_out_buf = metalalloc.alloc(data_out_buf_size)
        temp_buf_A = metalalloc.alloc(temp_buf_size)
        temp_buf_B = metalalloc.alloc(temp_buf_size)
        reduction_results_buf = metalalloc.alloc(reduction_results_buf_size)

        prog_copy_sum = MetalProgram(Device[Device.DEFAULT], "copy_sum", compile(copy_sum_kernel))
        st_custom = time.perf_counter()

        # Stage 1 & 2: Find Global Max (no change from _sched_fusion_5, no exp involved)
        if n > 0:
            local_size_max1 = reduce_fat_ls
            items_per_thread_max1 = (n + max_workgroups * local_size_max1[0] - 1) // (max_workgroups * local_size_max1[0]) or 1
            items_per_workgroup_max1 = items_per_thread_max1 * local_size_max1[0]
            num_wg_stage1 = (n + items_per_workgroup_max1 - 1) // items_per_workgroup_max1
            if num_wg_stage1 > max_workgroups: num_wg_stage1 = max_workgroups

            params_max_stage1 = {"n": f"{n}UL", "local_size": local_size_max1, "items_per_workgroup": items_per_workgroup_max1}
            prog_max_stage1 = MetalProgram(Device[Device.DEFAULT], "generic_reduce_max", compile(generic_reduce_max.format(**params_max_stage1)))
            prog_max_stage1(temp_buf_A, data_in_buf, global_size=[num_wg_stage1, 1, 1], local_size=local_size_max1)

            # Stage 2: Recursive reduction on partial maxes in temp_buf_A
            src_buf_max, current_size_max = temp_buf_A, num_wg_stage1
            i_max = 0
            temp_bufs_max = [temp_buf_B, temp_buf_A]
            while current_size_max > 1:
                local_size_thin = reduce_thin_ls
                num_wg_thin = (current_size_max + local_size_thin[0] - 1) // local_size_thin[0]
                params_max_thin = {"n": f"{current_size_max}UL", "local_size": local_size_thin}
                prog_max_thin = MetalProgram(Device[Device.DEFAULT], "global_reduce_max", compile(global_reduce_max.format(**params_max_thin)))
                dest_buf_max = temp_bufs_max[i_max % 2]
                prog_max_thin(dest_buf_max, src_buf_max, global_size=[num_wg_thin, 1, 1], local_size=local_size_thin)
                src_buf_max = dest_buf_max
                current_size_max = num_wg_thin
                i_max += 1
            final_max_buf = src_buf_max
            metalalloc._transfer(reduction_results_buf, final_max_buf, 4, src_dev=metalalloc.dev, dest_dev=metalalloc.dev)

        # Stage 3 & 4: Find Global Sum (using fast_exp for first stage)
        num_wg_stage3 = 0
        if n > 0:
            # Stage 3: First-level reduction for sum (data -> partial sums)
            local_size_sum1 = reduce_fat_ls
            items_per_thread_sum1 = (n + max_workgroups * local_size_sum1[0] - 1) // (max_workgroups * local_size_sum1[0]) or 1
            items_per_workgroup_sum1 = items_per_thread_sum1 * local_size_sum1[0]
            num_wg_stage3 = (n + items_per_workgroup_sum1 - 1) // items_per_workgroup_sum1
            if num_wg_stage3 > max_workgroups: num_wg_stage3 = max_workgroups
            params_sum_stage3 = {"n": f"{n}UL", "local_size": local_size_sum1, "items_per_workgroup": items_per_workgroup_sum1}
            prog_sum_stage3 = MetalProgram(Device[Device.DEFAULT], "generic_reduce_sum", compile(generic_reduce_sum_fast_exp.format(**params_sum_stage3)))
            prog_sum_stage3(temp_buf_A, data_in_buf, reduction_results_buf, global_size=[num_wg_stage3, 1, 1], local_size=local_size_sum1)

        # Stage 4: Recursive reduction on partial sums (no exp, same as _sched_fusion_5)
        if num_wg_stage3 > 0:
            src_buf_sum, current_size_sum = temp_buf_A, num_wg_stage3
            i_sum = 0
            temp_bufs_sum = [temp_buf_B, temp_buf_A]
            while current_size_sum > 1:
                local_size_thin = reduce_thin_ls
                num_wg_thin = (current_size_sum + local_size_thin[0] - 1) // local_size_thin[0]
                params_sum_thin = {"n": f"{current_size_sum}UL", "local_size": local_size_thin}
                prog_sum_thin = MetalProgram(Device[Device.DEFAULT], "global_sum", compile(global_sum.format(**params_sum_thin)))
                dest_buf_sum = temp_bufs_sum[i_sum % 2]
                prog_sum_thin(dest_buf_sum, src_buf_sum, global_size=[num_wg_thin, 1, 1], local_size=local_size_thin)
                src_buf_sum = dest_buf_sum
                current_size_sum = num_wg_thin
                i_sum += 1
            # After reduction, src_buf_sum holds the buffer with the final sum. Copy it.
            prog_copy_sum(reduction_results_buf, src_buf_sum, global_size=[1,1,1], local_size=[1,1,1])

        # Stage 5: Final division (using fast_exp)
        if n > 0:
            final_div_gs = min(max_workgroups, (n + div_ls[0] - 1) // div_ls[0])
            prog_final_div = MetalProgram(Device[Device.DEFAULT], "final_division", compile(final_division_fast_exp.format(local_size=div_ls, global_size=[final_div_gs,1,1], n=f"{n}UL")))
            prog_final_div(data_out_buf, data_in_buf, reduction_results_buf, global_size=[final_div_gs, 1, 1], local_size=div_ls)

        Device[Device.DEFAULT].synchronize()
        custom_time = time.perf_counter() - st_custom

        d0 = np.empty((n), dtype=np.float32)
        if n > 0:
            metalalloc._copyout(flat_mv(d0.data), data_out_buf)

        metalalloc.free(data_in_buf, data_in_buf_size)
        metalalloc.free(data_out_buf, data_out_buf_size)
        metalalloc.free(temp_buf_A, temp_buf_size)
        metalalloc.free(temp_buf_B, temp_buf_size)
        metalalloc.free(reduction_results_buf, reduction_results_buf_size)

        return custom_time, d0

    def _run_recursive_reduction(self, n_items: int, src_buf, kernel_name: str, kernel_src_template: str, result_buf = None):
        """Helper to run a generic, multi-stage recursive reduction."""
        reduce_ls = [256, 1, 1]
        max_wg_recursive = 16384
        temp_reduction_buf_A = metalalloc.alloc(max_wg_recursive * 4)
        temp_reduction_buf_B = metalalloc.alloc(max_wg_recursive * 4)
        temp_bufs = [temp_reduction_buf_A, temp_reduction_buf_B]

        current_size = n_items
        i = 0
        src_buf_recursive = src_buf
        while current_size > 1:
            num_wg_reduce = (current_size + reduce_ls[0] - 1) // reduce_ls[0]
            prog_reduce = MetalProgram(Device[Device.DEFAULT], kernel_name, compile(kernel_src_template.format(n=f"{current_size}UL", local_size=reduce_ls)))
            # On the final step, write to the final result buffer if provided, otherwise to the last temp buffer.
            is_final_step = (num_wg_reduce == 1)
            dest_buf = (result_buf if is_final_step and result_buf is not None else temp_bufs[i % 2])
            prog_reduce(dest_buf, src_buf_recursive, global_size=[num_wg_reduce, 1, 1], local_size=reduce_ls)
            src_buf_recursive = dest_buf
            current_size = num_wg_reduce
            i += 1

        if n_items == 1 and result_buf is not None: # Handle case where only one partial result was produced
            metalalloc._transfer(result_buf, src_buf_recursive, 4, src_dev=metalalloc.dev, dest_dev=metalalloc.dev)
        elif current_size == 1 and result_buf is not None and src_buf_recursive != result_buf: # Result ended up in a temp buffer
            metalalloc._transfer(result_buf, src_buf_recursive, 4, src_dev=metalalloc.dev, dest_dev=metalalloc.dev)

        metalalloc.free(temp_reduction_buf_A, max_wg_recursive * 4)
        metalalloc.free(temp_reduction_buf_B, max_wg_recursive * 4)
        return src_buf_recursive

    def _run_recursive_reduction_vector(self, n_items: int, src_buf, kernel_name: str, kernel_src_template: str, result_buf=None):
        """Helper to run a multi-stage recursive reduction using vectorized kernels that handle unaligned tails."""
        reduce_ls = [256, 1, 1]
        max_wg_recursive = 16384
        temp_reduction_buf_A = metalalloc.alloc(max_wg_recursive * 4)
        temp_reduction_buf_B = metalalloc.alloc(max_wg_recursive * 4)
        temp_bufs = [temp_reduction_buf_A, temp_reduction_buf_B]

        current_size = n_items
        i = 0
        src_buf_recursive = src_buf
        while current_size > 1:
            num_threads = (current_size + 3) // 4
            num_wg_reduce = (num_threads + reduce_ls[0] - 1) // reduce_ls[0]
            prog_reduce = MetalProgram(Device[Device.DEFAULT], kernel_name, compile(kernel_src_template.format(n=f"{current_size}UL", local_size=reduce_ls)))

            is_final_step = (num_wg_reduce == 1)
            dest_buf = (result_buf if is_final_step and result_buf is not None else temp_bufs[i % 2])
            prog_reduce(dest_buf, src_buf_recursive, global_size=[num_wg_reduce, 1, 1], local_size=reduce_ls)

            src_buf_recursive = dest_buf
            current_size = num_wg_reduce
            i += 1

        if n_items == 1 and result_buf is not None:
            metalalloc._transfer(result_buf, src_buf_recursive, 4, src_dev=metalalloc.dev, dest_dev=metalalloc.dev)
        elif current_size == 1 and result_buf is not None and src_buf_recursive != result_buf:
            metalalloc._transfer(result_buf, src_buf_recursive, 4, src_dev=metalalloc.dev, dest_dev=metalalloc.dev)

        metalalloc.free(temp_reduction_buf_A, max_wg_recursive * 4)
        metalalloc.free(temp_reduction_buf_B, max_wg_recursive * 4)
        return src_buf_recursive

    def _sched_fusion_5_opt_base(self, n: int, d1: np.ndarray, first_stage_max_kernel: str, first_stage_sum_kernel: str, final_div_kernel: str,
                                   max_kernel_name: str, sum_kernel_name: str, div_kernel_name: str, is_vectorized: bool, use_simd: bool):
        if n == 0: return 0.0, np.array([], dtype=np.float32)

        # 1. Tuning parameters
        items_per_block = 4096
        local_size_stage1 = [256, 1, 1]
        simd_size = 32

        # 2. Pad input data
        padded_n = ((n + items_per_block - 1) // items_per_block) * items_per_block
        d1_padded = np.full((padded_n), -np.inf, dtype=np.float32)
        d1_padded[:n] = d1

        # 3. Buffer allocation
        num_workgroups_stage1 = padded_n // items_per_block
        data_in_buf_size = padded_n * 4
        data_out_buf_size = padded_n * 4
        reduction_results_buf_size = 2 * 4
        partial_results_buf_size = num_workgroups_stage1 * 4

        data_in_buf = metalalloc.alloc(data_in_buf_size); metalalloc._copyin(data_in_buf, d1_padded.tobytes())
        data_out_buf = metalalloc.alloc(data_out_buf_size)
        reduction_results_buf = metalalloc.alloc(reduction_results_buf_size)
        partial_max_buf = metalalloc.alloc(partial_results_buf_size)
        partial_sum_buf = metalalloc.alloc(partial_results_buf_size)

        # 4. Compile kernels
        prog_copy_sum = MetalProgram(Device[Device.DEFAULT], "copy_sum", compile(copy_sum_kernel))
        prog_params_stage1 = {"items_per_block": items_per_block, "local_size": local_size_stage1}
        if use_simd: prog_params_stage1["threads_per_simdgroup"] = simd_size

        prog_stage1_max = MetalProgram(Device[Device.DEFAULT], max_kernel_name, compile(first_stage_max_kernel.format(**prog_params_stage1)))
        prog_stage1_sum = MetalProgram(Device[Device.DEFAULT], sum_kernel_name, compile(first_stage_sum_kernel.format(**prog_params_stage1)))

        st_custom = time.perf_counter()

        # Determine which recursive reduction to use
        if is_vectorized:
            recursive_reduce_func = self._run_recursive_reduction_vector
            rec_max_kernel, rec_sum_kernel = global_reduce_max_vector_kernel, global_reduce_sum_vector_kernel
            rec_max_name, rec_sum_name = "global_reduce_max_vector", "global_reduce_sum_vector"
        else: # For register-based (SIMD) or other scalar approaches
            recursive_reduce_func = self._run_recursive_reduction
            rec_max_kernel, rec_sum_kernel = global_reduce_max, global_sum
            rec_max_name, rec_sum_name = "global_reduce_max", "global_sum"

        # Stage 1: First-level max reduction
        prog_stage1_max(partial_max_buf, data_in_buf, global_size=[num_workgroups_stage1, 1, 1], local_size=local_size_stage1)
        # Stage 2: Recursive global max reduction
        recursive_reduce_func(num_workgroups_stage1, partial_max_buf, rec_max_name, rec_max_kernel, result_buf=reduction_results_buf)

        # Stage 3: First-level sum reduction
        prog_stage1_sum(partial_sum_buf, data_in_buf, reduction_results_buf, global_size=[num_workgroups_stage1, 1, 1], local_size=local_size_stage1)
        # Stage 4: Recursive global sum reduction
        final_sum_buf = recursive_reduce_func(num_workgroups_stage1, partial_sum_buf, rec_sum_name, rec_sum_kernel)
        prog_copy_sum(reduction_results_buf, final_sum_buf, global_size=[1,1,1], local_size=[1,1,1])

        # Stage 5: Final division
        if is_vectorized:
            ITEMS_PER_THREAD_DIV = 8; local_size_stage3 = [128, 1, 1]
            num_chunks_div = padded_n // ITEMS_PER_THREAD_DIV
            grid_size_stage3 = (num_chunks_div + local_size_stage3[0] - 1) // local_size_stage3[0]
            prog_final_div = MetalProgram(Device[Device.DEFAULT], div_kernel_name, compile(final_div_kernel.format(local_size=local_size_stage3, ITEMS_PER_THREAD=ITEMS_PER_THREAD_DIV)))
            prog_final_div(data_out_buf, data_in_buf, reduction_results_buf, global_size=[grid_size_stage3, 1, 1], local_size=local_size_stage3)
        else:
            max_workgroups = 16384; local_size_stage3 = [512, 1, 1]
            final_div_gs = min(max_workgroups, (n + local_size_stage3[0] - 1) // local_size_stage3[0])
            prog_final_div = MetalProgram(Device[Device.DEFAULT], div_kernel_name, compile(final_div_kernel.format(local_size=local_size_stage3, global_size=[final_div_gs,1,1], n=f"{n}UL")))
            prog_final_div(data_out_buf, data_in_buf, reduction_results_buf, global_size=[final_div_gs, 1, 1], local_size=local_size_stage3)

        Device[Device.DEFAULT].synchronize()
        custom_time = time.perf_counter() - st_custom

        d0_padded = np.empty((padded_n), dtype=np.float32)
        metalalloc._copyout(flat_mv(d0_padded.data), data_out_buf)
        d0 = d0_padded[:n]

        # Free all allocated buffers
        metalalloc.free(data_in_buf, data_in_buf_size); metalalloc.free(data_out_buf, data_out_buf_size)
        metalalloc.free(partial_max_buf, partial_results_buf_size); metalalloc.free(partial_sum_buf, partial_results_buf_size)
        metalalloc.free(reduction_results_buf, reduction_results_buf_size)
        return custom_time, d0

    def _sched_fusion_5_register(self, n: int, d1: np.ndarray) -> tuple[float, np.ndarray]:
        return self._sched_fusion_5_opt_base(n, d1, fusion5_reg_reduce_max_kernel, fusion5_reg_reduce_sum_kernel, final_division,
                                            "fusion5_reg_reduce_max", "fusion5_reg_reduce_sum", "final_division",
                                            is_vectorized=False, use_simd=True)

    def _sched_fusion_5_vector(self, n: int, d1: np.ndarray) -> tuple[float, np.ndarray]:
        return self._sched_fusion_5_opt_base(n, d1, fusion5_vec_reduce_max_kernel, fusion5_vec_reduce_sum_kernel, global_div_opt,
                                            "fusion5_vec_reduce_max", "fusion5_vec_reduce_sum", "global_div_opt",
                                            is_vectorized=True, use_simd=False)

    def _sched_fusion_3(self, n: int, d1: np.ndarray) -> tuple[float, np.ndarray]:
        if n == 0: return 0.0, np.array([], dtype=np.float32)
        local_size_stage1 = [256, 1, 1]
        n = int(n)
        num_workgroup = (n + local_size_stage1[0] - 1) // local_size_stage1[0]
        padded_n = num_workgroup * local_size_stage1[0]

        d1_padded = np.full((padded_n), -np.inf, dtype=np.float32)
        d1_padded[:n] = d1

        # --- START: REFACTORED BUFFER SIZING ---
        max_workgroups_for_reduce = 16384 # Cap for temporary buffer size
        data_in_buf_size = padded_n * 4
        data_out_buf_size = padded_n * 4
        intermediate_buf_size = num_workgroup * 2 * 4 # This can still be large, but it's read-only after creation.
        reduction_results_buf_size = 2 * 4
        adjusted_sums_buf_size = num_workgroup * 4
        # Ping-pong buffers are now capped to a reasonable size.
        temp_reduction_buf_size = max_workgroups_for_reduce * 4
        # --- END: REFACTORED BUFFER SIZING ---

        data_in_buf = metalalloc.alloc(data_in_buf_size)
        metalalloc._copyin(data_in_buf, d1_padded.tobytes())
        data_out_buf = metalalloc.alloc(data_out_buf_size)
        intermediate_buf = metalalloc.alloc(intermediate_buf_size)
        reduction_results_buf = metalalloc.alloc(reduction_results_buf_size)
        adjusted_sums_buf = metalalloc.alloc(adjusted_sums_buf_size)
        # --- START: REFACTORED BUFFER ALLOCATION ---
        temp_reduction_buf_A = metalalloc.alloc(temp_reduction_buf_size)
        temp_reduction_buf_B = metalalloc.alloc(temp_reduction_buf_size)
        # --- END: REFACTORED BUFFER ALLOCATION ---

        prog_local_max_and_sum = MetalProgram(Device[Device.DEFAULT], "local_max_and_sum",
                                              compile(local_max_and_sum.format(local_size=local_size_stage1, num_workgroup=num_workgroup)))
        prog_final_div = MetalProgram(Device[Device.DEFAULT], "global_div",
                                      compile(global_div.format(local_size=local_size_stage1)))
        prog_adjust_sums = MetalProgram(Device[Device.DEFAULT], "adjust_sums",
                                        compile(adjust_sums_kernel.format(num_workgroup=num_workgroup)))
        prog_copy_sum = MetalProgram(Device[Device.DEFAULT], "copy_sum", compile(copy_sum_kernel))

        st_custom = time.perf_counter()

        # Kernel 1: local reduction of max and sum for each workgroup
        prog_local_max_and_sum(intermediate_buf, data_in_buf,
                               global_size=[num_workgroup, 1, 1], local_size=local_size_stage1)

        # --- START: REFACTORED ITERATIVE REDUCTION LOGIC ---

        # Stage 2a: Recursively reduce the local max values to find the global max.
        reduce_ls = [256, 1, 1]
        current_size = num_workgroup
        # The local maxes are in the first half of intermediate_buf.
        src_buf_max = intermediate_buf

        if current_size > 1:
            temp_bufs = [temp_reduction_buf_A, temp_reduction_buf_B]
            i = 0
            while current_size > 1:
                num_wg_reduce = (current_size + reduce_ls[0] - 1) // reduce_ls[0]
                prog_reduce = MetalProgram(Device[Device.DEFAULT], "global_reduce_max", compile(global_reduce_max.format(n=f"{current_size}UL", local_size=reduce_ls)))
                # On the final reduction step, write directly to the results buffer
                dest_buf = reduction_results_buf if num_wg_reduce == 1 else temp_bufs[i % 2]
                prog_reduce(dest_buf, src_buf_max, global_size=[num_wg_reduce, 1, 1], local_size=reduce_ls)
                src_buf_max = dest_buf
                current_size = num_wg_reduce
                i += 1
        else: # If only one workgroup to begin with, just copy its max
            metalalloc._transfer(reduction_results_buf, src_buf_max, 4, src_dev=metalalloc.dev, dest_dev=metalalloc.dev)

        # Stage 2b: Adjust local sums using the global max.
        prog_adjust_sums(adjusted_sums_buf, intermediate_buf, reduction_results_buf,
                         global_size=[num_workgroup, 1, 1], local_size=[min(num_workgroup, 1024), 1, 1])

        # Stage 2c: Recursively reduce the adjusted sums to find the global sum.
        current_size = num_workgroup
        src_buf_sum = adjusted_sums_buf

        if current_size > 1:
            temp_bufs = [temp_reduction_buf_A, temp_reduction_buf_B]
            i = 0
            while current_size > 1:
                num_wg_reduce = (current_size + reduce_ls[0] - 1) // reduce_ls[0]
                prog_reduce = MetalProgram(Device[Device.DEFAULT], "global_sum", compile(global_sum.format(n=f"{current_size}UL", local_size=reduce_ls)))

                # Simple ping-pong: destination is always the other buffer.
                dest_buf = temp_bufs[i % 2]
                prog_reduce(dest_buf, src_buf_sum, global_size=[num_wg_reduce, 1, 1], local_size=reduce_ls)

                # The new source is the destination of the last run.
                src_buf_sum = dest_buf
                current_size = num_wg_reduce
                i += 1

        # After the loop (or if it was skipped), src_buf_sum holds the buffer with the final sum.
        final_sum_buf = src_buf_sum

        # Copy the single final sum into the second slot of the results buffer.
        prog_copy_sum(reduction_results_buf, final_sum_buf, global_size=[1,1,1], local_size=[1,1,1])

        # --- END: REFACTORED ITERATIVE REDUCTION LOGIC ---

        # Kernel 3: final calculation using global max and global sum.
        prog_final_div(data_out_buf, data_in_buf, reduction_results_buf,
                       global_size=[num_workgroup, 1, 1], local_size=local_size_stage1)

        Device[Device.DEFAULT].synchronize()
        custom_time = time.perf_counter() - st_custom

        d0_padded = np.empty((padded_n), dtype=np.float32)
        metalalloc._copyout(flat_mv(d0_padded.data), data_out_buf)
        d0 = d0_padded[:n]

        metalalloc.free(data_in_buf, data_in_buf_size)
        metalalloc.free(data_out_buf, data_out_buf_size)
        metalalloc.free(intermediate_buf, intermediate_buf_size)
        metalalloc.free(reduction_results_buf, reduction_results_buf_size)
        metalalloc.free(adjusted_sums_buf, adjusted_sums_buf_size)
        metalalloc.free(temp_reduction_buf_A, temp_reduction_buf_size)
        metalalloc.free(temp_reduction_buf_B, temp_reduction_buf_size)

        return custom_time, d0

    def test_base(self):
        n = random.randint(0, 2**14)
        self._driver("base", n, self._sched_base)

    def test_fusion_5(self):
        n = random.randint(0, 2**29)
        self._driver("fusion 5", n, self._sched_fusion_5)

    def test_fusion_3(self):
        n = random.randint(0, 2**29)
        self._driver("fusion 3", n, self._sched_fusion_3)

    def test_fusion_5_tune(self):
        n = random.randint(0, 2**24)
        rng = np.random.default_rng()
        d1 = rng.standard_normal(size=(n), dtype=np.float32)

        param = [2 ** i for i in range(5, 10)]
        param_configs = {
            "reduce_fat_ls": param,
            "reduce_thin_ls": param,
            "div_ls": param,
        }

        # Run the tuning process.
        best_time, best_params = fine_tune_scheduler(
            self._sched_fusion_5,
            n,
            param_configs,
            verbose=True
        )

        # Run one last time with the best parameters to get the result for verification.
        best_params_kwargs = {k: [v, 1, 1] for k, v in best_params.items()}
        _, result = self._sched_fusion_5(n, d1, **best_params_kwargs)

        # Get the ground truth from tinygrad for verification.
        @TinyJit
        def tiny_jit(t: Tensor) -> Tensor:
            return t.softmax().realize()
        tiny_softmax = tiny_jit(Tensor(d1)).numpy()

        # Verify the numerical correctness of the tuned result.
        np.testing.assert_allclose(result, tiny_softmax, rtol=1e-6, atol=1e-6)
        print(f"\nVerification successful for tuned 'fusion 5' with best params={best_params} ({best_time:.4f}s)")

    def test_fusion_5_register(self):
        n = random.randint(0, 2**29)
        self._driver("fusion 5 register", n, self._sched_fusion_5_register)

    def test_fusion_5_vector(self):
        n = random.randint(0, 2**29)
        self._driver("fusion 5 vector", n, self._sched_fusion_5_vector)

    def test_fusion_5_fast_exp(self):
        n = random.randint(0, 2**29)
        self._driver("fusion 5 fast exp", n, self._sched_fusion_5_fast_exp, rtol=1e-5, atol=1e-5)


def plot_results(results: dict):
    """Generates and saves a plot of performance vs. input size."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    for name, data in results.items():
        if data['n']:
            ax.plot(data['n'], data['gflops'], label=name, marker='o', linestyle='-')

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Input Size (n)')
    ax.set_ylabel('Performance (GFLOPS)')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best')

    plot_filename = "benchmark.png"
    plt.tight_layout()
    plt.savefig(plot_filename)
    print(f"\nBenchmark finished. Plot saved to {plot_filename}")

def run_benchmark_and_plot():
    if Device.DEFAULT != "METAL":
        print("Benchmark mode requires Metal backend. Skipping.")
        return

    sm_instance = Softmax()
    param_configs = {
        "reduce_fat_ls": [2 ** i for i in range(5, 11)],
        "reduce_thin_ls": [2 ** i for i in range(5, 11)],
        "div_ls": [2 ** i for i in range(5, 11)],
    }
    n_values = [2**i for i in range(10, 31)]
    rng = np.random.default_rng()

    # --- Stage 1: Pre-computation to find optimal parameters for each 'n' ---
    print("\n--- STAGE 1: Fine-tuning 'fusion 5' to find optimal parameters. This may take a while... ---")
    tuned_params_cache = {}
    for n in n_values:
        try:
            _, best_params = fine_tune_scheduler(
                sm_instance._sched_fusion_5, n, param_configs, verbose=False
            )
            tuned_params_cache[n] = best_params
            print(f"  [n={n:>10}] Found best params: {best_params}")
        except Exception as e:
            print(f"  [n={n:>10}] Tuning failed: {e}")
            tuned_params_cache[n] = None
    print("--- STAGE 1 Complete ---\n")

    implementations = {
        "fusion 5": sm_instance._sched_fusion_5,
        "fusion 5 tuned": None,
        "fusion 5 fast exp": sm_instance._sched_fusion_5_fast_exp,
        "fusion 5 register": sm_instance._sched_fusion_5_register,
        "fusion 5 vector": sm_instance._sched_fusion_5_vector,
        "fusion 3": sm_instance._sched_fusion_3,
    }

    results = {name: {'n': [], 'gflops': []} for name in ["tinygrad"] + list(implementations.keys())}
    NUM_RUNS = 2**1

    for n in n_values:
        print(f"\n--- benchmark for n={n} ({n/1e6:.2f}M elements) ---")

        @TinyJit
        def tiny_jit(t: Tensor) -> Tensor:
            return t.softmax().realize()

        # 1. Run tinygrad to get baseline performance and op count
        ops = -1
        try:
            timed_tensors = [Tensor(rng.standard_normal(size=(n), dtype=np.float32)) for _ in range(NUM_RUNS)]

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
            if ops > 0 and tiny_time > 0:
                gflops = ops / tiny_time / 1e9
                results['tinygrad']['n'].append(n); results['tinygrad']['gflops'].append(gflops)
                print(f"{'tinygrad':<22}: {gflops:>7.2f} GFLOPS (median time: {tiny_time:.4f}s)")
        except Exception as e:
            print(f"tinygrad failed for n={n}: {e}")

        if ops <= 0:
            print("Skipping custom kernels for this size as op count is unknown.")
            continue

        for name, executor_func in implementations.items():
            executor = executor_func
            if name == "fusion 5 tuned":
                best_params = tuned_params_cache.get(n)
                if best_params is None:
                    print(f"{name:<22}: Skipped (tuning failed for this size)")
                    continue
                # Create the executor on-the-fly with the cached optimal parameters
                best_params_kwargs = {k: [v, 1, 1] for k, v in best_params.items()}
                executor = lambda n_inner, d1_inner: sm_instance._sched_fusion_5(n_inner, d1_inner, **best_params_kwargs)

            try:
                d1_warmup = rng.standard_normal(size=(n), dtype=np.float32)
                executor(n, d1_warmup); Device[Device.DEFAULT].synchronize(); del d1_warmup

                custom_times = []
                for _ in range(NUM_RUNS):
                    d1_timed = rng.standard_normal(size=(n), dtype=np.float32)
                    run_time, _ = executor(n, d1_timed)
                    custom_times.append(run_time)
                    del d1_timed
                median_time = np.median(custom_times)

                if median_time > 0 and not np.isnan(median_time):
                    gflops = ops / median_time / 1e9
                    results[name]['n'].append(n); results[name]['gflops'].append(gflops)
                    params_str = ""
                    if name == "fusion 5 tuned":
                        params_str = f"(params: {tuned_params_cache.get(n)})"
                    print(f"{name:<22}: {gflops:>7.2f} GFLOPS (median time: {median_time:.4f}s) {params_str}".rstrip())
            except Exception as e:
                print(f"{name:<22}: Failed with exception: {e}")

    plot_results(results)

if __name__ == "__main__":
    if os.environ.get("BENCHMARK"):
        run_benchmark_and_plot()
    else:
        unittest.main(verbosity=2)
