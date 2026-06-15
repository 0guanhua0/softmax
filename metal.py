import time
from functools import cache

import numpy as np
from tinygrad.device import Device
from tinygrad.helpers import flat_mv
from tinygrad.runtime.ops_metal import MetalAllocator, MetalProgram

import perf


@cache
def compile(src: str) -> bytes:
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
        if (i < {n}) {{
            p_max = fmax(p_max, data_in[i]);
        }}
    }}

    threadgroup float local_data[{local_size[0]}];
    local_data[lid0] = p_max;
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

generic_reduce_sum = """
#include <metal_stdlib>
using namespace metal;

kernel void generic_reduce_sum(device float *data_out, device const float *data_in, device const float* global_max_val, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    uint gid0 = gid.x;
    uint lid0 = lid.x;
    float max_val = *global_max_val;

    float p_sum = 0.0f;
    size_t chunk_start = (size_t)gid0 * {items_per_workgroup};
    size_t chunk_end = chunk_start + {items_per_workgroup};

    for (size_t i = chunk_start + lid0; i < chunk_end; i += {local_size[0]}) {{
        if (i < {n}) {{
            p_sum += exp(data_in[i] - max_val);
        }}
    }}

    threadgroup float local_data[{local_size[0]}];
    local_data[lid0] = p_sum;
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

    for (size_t i = block_start_idx + lid.x; i < block_end_idx; i += {local_size[0]}) {{
        thread_max = fmax(thread_max, data_in[i]);
    }}

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

    for (size_t i = block_start_idx + lid.x; i < block_end_idx; i += {local_size[0]}) {{
        thread_sum += exp(data_in[i] - global_max);
    }}

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

    for (uint i = lid.x; i < ({items_per_block} / 4); i += {local_size[0]}) {{
        float4 val = data_in_vec[block_start_vec_idx + i];
        thread_max = fmax(thread_max, fmax(fmax(val.x, val.y), fmax(val.z, val.w)));
    }}

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

    size_t block_start_vec_idx = (size_t)gid.x * ({items_per_block} / 4);
    device const float4* data_in_vec = (device const float4*)data_in;
    float4 global_max_vec = float4(global_max);

    for (uint i = lid.x; i < ({items_per_block} / 4); i += {local_size[0]}) {{
        float4 val = data_in_vec[block_start_vec_idx + i] - global_max_vec;
        float4 exp_val;
        exp_val.x = exp(val.x);
        exp_val.y = exp(val.y);
        exp_val.z = exp(val.z);
        exp_val.w = exp(val.w);
        thread_sum += (exp_val.x + exp_val.y + exp_val.z + exp_val.w);
    }}

    threadgroup float local_data[{local_size[0]}];
    local_data[lid.x] = thread_sum;
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

global_reduce_max_vector_kernel = """
#include <metal_stdlib>
using namespace metal;

kernel void global_reduce_max_vector(device float *data_out, device const float *data_in, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    threadgroup float local_data[{local_size[0]}];
    float p_max = -FLT_MAX;
    size_t global_thread_idx = (size_t)gid.x * {local_size[0]} + lid.x;
    size_t start_idx = global_thread_idx * 4;

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
    uint gid0 = gid.x;
    uint lid0 = lid.x;
    size_t global_idx = (size_t)gid0 * {local_size[0]} + lid0;

    threadgroup float local_values[{local_size[0]}];
    threadgroup float temp_storage[{local_size[0]}];

    local_values[lid0] = data_in[global_idx];
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

    local_values[lid0] = exp(local_values[lid0] - local_max);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = {local_size[0]} / 2; stride > 0; stride /= 2) {{
        if (lid0 < stride) {{
            local_values[lid0] += local_values[lid0 + stride];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float local_sum = local_values[0];

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

    size_t chunk_idx = (size_t)gid.x * {local_size[0]} + lid.x;
    size_t start_elem_idx = chunk_idx * {ITEMS_PER_THREAD};

    device const float4* data_in_vec = (device const float4*)(data_in0 + start_elem_idx);
    device float4* data_out_vec = (device float4*)(data_out + start_elem_idx);
    float4 max_val_vec = float4(max_val);

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


def _sched_base(n: int, data: np.ndarray) -> tuple[float, np.ndarray]:
    global_size = [n, 1, 1]
    prog = MetalProgram(
        Device[Device.DEFAULT],
        "base",
        compile(base.format(global_size=global_size)),
    )

    data_out_size = n * 4
    data_in_size = n * 4
    data_out = metalalloc.alloc(data_out_size)
    data_in = metalalloc.alloc(data_in_size)
    metalalloc._copyin(data_in, data.tobytes())

    st_custom = time.perf_counter()
    prog(data_out, data_in, global_size=global_size, local_size=[1, 1, 1], wait=True)
    custom_time = time.perf_counter() - st_custom

    d0 = np.empty((n), dtype=np.float32)
    metalalloc._copyout(flat_mv(d0.data), data_out)

    metalalloc.free(data_out, data_out_size)
    metalalloc.free(data_in, data_in_size)

    return custom_time, d0


def _sched_fusion_5(n: int, data: np.ndarray, **kwargs) -> tuple[float, np.ndarray]:
    default_reduce_ls = [256, 1, 1]
    default_div_ls = [256, 1, 1]

    reduce_fat_ls = kwargs.get("reduce_fat_ls", default_reduce_ls)
    reduce_thin_ls = kwargs.get("reduce_thin_ls", default_reduce_ls)
    div_ls = kwargs.get("div_ls", default_div_ls)

    max_workgroups = 16384

    data_in_buf_size = n * 4
    data_out_buf_size = n * 4
    temp_buf_size = max_workgroups * 4
    reduction_results_buf_size = 2 * 4

    data_in_buf = metalalloc.alloc(data_in_buf_size)
    metalalloc._copyin(data_in_buf, data.tobytes())
    data_out_buf = metalalloc.alloc(data_out_buf_size)
    temp_buf_A = metalalloc.alloc(temp_buf_size)
    temp_buf_B = metalalloc.alloc(temp_buf_size)
    reduction_results_buf = metalalloc.alloc(reduction_results_buf_size)

    prog_copy_sum = MetalProgram(
        Device[Device.DEFAULT], "copy_sum", compile(copy_sum_kernel)
    )
    st_custom = time.perf_counter()

    if n > 0:
        local_size_max1 = reduce_fat_ls
        items_per_thread_max1 = (n + max_workgroups * local_size_max1[0] - 1) // (
            max_workgroups * local_size_max1[0]
        ) or 1
        items_per_workgroup_max1 = items_per_thread_max1 * local_size_max1[0]
        num_wg_stage1 = (n + items_per_workgroup_max1 - 1) // items_per_workgroup_max1
        if num_wg_stage1 > max_workgroups:
            num_wg_stage1 = max_workgroups

        params_max_stage1 = {
            "n": f"{n}UL",
            "local_size": local_size_max1,
            "items_per_workgroup": items_per_workgroup_max1,
        }
        prog_max_stage1 = MetalProgram(
            Device[Device.DEFAULT],
            "generic_reduce_max",
            compile(generic_reduce_max.format(**params_max_stage1)),
        )
        prog_max_stage1(
            temp_buf_A,
            data_in_buf,
            global_size=[num_wg_stage1, 1, 1],
            local_size=local_size_max1,
        )

        _run_recursive_reduction(
            num_wg_stage1,
            temp_buf_A,
            "global_reduce_max",
            global_reduce_max,
            result_buf=reduction_results_buf,
            reduce_ls=reduce_thin_ls,
            temp_buf_A=temp_buf_A,
            temp_buf_B=temp_buf_B,
        )

    num_wg_stage3 = 0
    if n > 0:
        local_size_sum1 = reduce_fat_ls
        items_per_thread_sum1 = (n + max_workgroups * local_size_sum1[0] - 1) // (
            max_workgroups * local_size_sum1[0]
        ) or 1
        items_per_workgroup_sum1 = items_per_thread_sum1 * local_size_sum1[0]
        num_wg_stage3 = (n + items_per_workgroup_sum1 - 1) // items_per_workgroup_sum1
        if num_wg_stage3 > max_workgroups:
            num_wg_stage3 = max_workgroups
        params_sum_stage3 = {
            "n": f"{n}UL",
            "local_size": local_size_sum1,
            "items_per_workgroup": items_per_workgroup_sum1,
        }
        prog_sum_stage3 = MetalProgram(
            Device[Device.DEFAULT],
            "generic_reduce_sum",
            compile(generic_reduce_sum.format(**params_sum_stage3)),
        )
        prog_sum_stage3(
            temp_buf_A,
            data_in_buf,
            reduction_results_buf,
            global_size=[num_wg_stage3, 1, 1],
            local_size=local_size_sum1,
        )

    if num_wg_stage3 > 0:
        final_sum_buf = _run_recursive_reduction(
            num_wg_stage3,
            temp_buf_A,
            "global_sum",
            global_sum,
            reduce_ls=reduce_thin_ls,
            temp_buf_A=temp_buf_A,
            temp_buf_B=temp_buf_B,
        )
        prog_copy_sum(
            reduction_results_buf,
            final_sum_buf,
            global_size=[1, 1, 1],
            local_size=[1, 1, 1],
        )

    if n > 0:
        final_div_gs = min(max_workgroups, (n + div_ls[0] - 1) // div_ls[0])
        prog_final_div = MetalProgram(
            Device[Device.DEFAULT],
            "final_division",
            compile(
                final_division.format(
                    local_size=div_ls, global_size=[final_div_gs, 1, 1], n=f"{n}UL"
                )
            ),
        )
        prog_final_div(
            data_out_buf,
            data_in_buf,
            reduction_results_buf,
            global_size=[final_div_gs, 1, 1],
            local_size=div_ls,
        )

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


def _run_recursive_reduction(
    n_items: int,
    src_buf,
    kernel_name: str,
    kernel_src_template: str,
    result_buf=None,
    is_vectorized: bool = False,
    reduce_ls: list = None,
    temp_buf_A=None,
    temp_buf_B=None,
):
    if reduce_ls is None:
        reduce_ls = [256, 1, 1]

    own_alloc = False
    if temp_buf_A is None or temp_buf_B is None:
        max_wg_recursive = 16384
        temp_buf_A = metalalloc.alloc(max_wg_recursive * 4)
        temp_buf_B = metalalloc.alloc(max_wg_recursive * 4)
        own_alloc = True

    temp_bufs = (
        [temp_buf_B, temp_buf_A] if src_buf == temp_buf_A else [temp_buf_A, temp_buf_B]
    )

    current_size = n_items
    i = 0
    src_buf_recursive = src_buf
    while current_size > 1:
        num_threads = (current_size + 3) // 4 if is_vectorized else current_size
        num_wg_reduce = (num_threads + reduce_ls[0] - 1) // reduce_ls[0]
        prog_reduce = MetalProgram(
            Device[Device.DEFAULT],
            kernel_name,
            compile(
                kernel_src_template.format(n=f"{current_size}UL", local_size=reduce_ls)
            ),
        )

        is_final_step = num_wg_reduce == 1
        dest_buf = (
            result_buf if is_final_step and result_buf is not None else temp_bufs[i % 2]
        )
        prog_reduce(
            dest_buf,
            src_buf_recursive,
            global_size=[num_wg_reduce, 1, 1],
            local_size=reduce_ls,
        )

        src_buf_recursive = dest_buf
        current_size = num_wg_reduce
        i += 1

    if n_items == 1 and result_buf is not None:
        metalalloc._transfer(
            result_buf,
            src_buf_recursive,
            4,
            src_dev=metalalloc.dev,
            dest_dev=metalalloc.dev,
        )
    elif (
        current_size == 1 and result_buf is not None and src_buf_recursive != result_buf
    ):
        metalalloc._transfer(
            result_buf,
            src_buf_recursive,
            4,
            src_dev=metalalloc.dev,
            dest_dev=metalalloc.dev,
        )

    if own_alloc:
        metalalloc.free(temp_buf_A, max_wg_recursive * 4)
        metalalloc.free(temp_buf_B, max_wg_recursive * 4)
    return src_buf_recursive


def _sched_fusion_5_opt_base(
    n: int,
    data: np.ndarray,
    first_stage_max_kernel: str,
    first_stage_sum_kernel: str,
    final_div_kernel: str,
    max_kernel_name: str,
    sum_kernel_name: str,
    div_kernel_name: str,
    is_vectorized: bool,
    use_simd: bool,
):
    items_per_block = 4096
    local_size_stage1 = [256, 1, 1]
    simd_size = 32

    padded_n = ((n + items_per_block - 1) // items_per_block) * items_per_block
    data_padded = np.full((padded_n), -np.inf, dtype=np.float32)
    data_padded[:n] = data

    num_workgroups_stage1 = padded_n // items_per_block
    data_in_buf_size = padded_n * 4
    data_out_buf_size = padded_n * 4
    reduction_results_buf_size = 2 * 4
    partial_results_buf_size = num_workgroups_stage1 * 4

    data_in_buf = metalalloc.alloc(data_in_buf_size)
    metalalloc._copyin(data_in_buf, data_padded.tobytes())
    data_out_buf = metalalloc.alloc(data_out_buf_size)
    reduction_results_buf = metalalloc.alloc(reduction_results_buf_size)
    partial_max_buf = metalalloc.alloc(partial_results_buf_size)
    partial_sum_buf = metalalloc.alloc(partial_results_buf_size)

    prog_copy_sum = MetalProgram(
        Device[Device.DEFAULT], "copy_sum", compile(copy_sum_kernel)
    )
    prog_params_stage1 = {
        "items_per_block": items_per_block,
        "local_size": local_size_stage1,
    }
    if use_simd:
        prog_params_stage1["threads_per_simdgroup"] = simd_size

    prog_stage1_max = MetalProgram(
        Device[Device.DEFAULT],
        max_kernel_name,
        compile(first_stage_max_kernel.format(**prog_params_stage1)),
    )
    prog_stage1_sum = MetalProgram(
        Device[Device.DEFAULT],
        sum_kernel_name,
        compile(first_stage_sum_kernel.format(**prog_params_stage1)),
    )

    st_custom = time.perf_counter()

    if is_vectorized:
        rec_max_kernel, rec_sum_kernel = (
            global_reduce_max_vector_kernel,
            global_reduce_sum_vector_kernel,
        )
        rec_max_name, rec_sum_name = (
            "global_reduce_max_vector",
            "global_reduce_sum_vector",
        )
    else:
        rec_max_kernel, rec_sum_kernel = global_reduce_max, global_sum
        rec_max_name, rec_sum_name = "global_reduce_max", "global_sum"

    prog_stage1_max(
        partial_max_buf,
        data_in_buf,
        global_size=[num_workgroups_stage1, 1, 1],
        local_size=local_size_stage1,
    )

    _run_recursive_reduction(
        num_workgroups_stage1,
        partial_max_buf,
        rec_max_name,
        rec_max_kernel,
        result_buf=reduction_results_buf,
        is_vectorized=is_vectorized,
    )

    prog_stage1_sum(
        partial_sum_buf,
        data_in_buf,
        reduction_results_buf,
        global_size=[num_workgroups_stage1, 1, 1],
        local_size=local_size_stage1,
    )

    final_sum_buf = _run_recursive_reduction(
        num_workgroups_stage1,
        partial_sum_buf,
        rec_sum_name,
        rec_sum_kernel,
        is_vectorized=is_vectorized,
    )
    prog_copy_sum(
        reduction_results_buf,
        final_sum_buf,
        global_size=[1, 1, 1],
        local_size=[1, 1, 1],
    )

    if is_vectorized:
        ITEMS_PER_THREAD_DIV = 8
        local_size_stage3 = [128, 1, 1]
        num_chunks_div = padded_n // ITEMS_PER_THREAD_DIV
        grid_size_stage3 = (
            num_chunks_div + local_size_stage3[0] - 1
        ) // local_size_stage3[0]
        prog_final_div = MetalProgram(
            Device[Device.DEFAULT],
            div_kernel_name,
            compile(
                final_div_kernel.format(
                    local_size=local_size_stage3,
                    ITEMS_PER_THREAD=ITEMS_PER_THREAD_DIV,
                )
            ),
        )
        prog_final_div(
            data_out_buf,
            data_in_buf,
            reduction_results_buf,
            global_size=[grid_size_stage3, 1, 1],
            local_size=local_size_stage3,
        )
    else:
        max_workgroups = 16384
        local_size_stage3 = [512, 1, 1]
        final_div_gs = min(
            max_workgroups, (n + local_size_stage3[0] - 1) // local_size_stage3[0]
        )
        prog_final_div = MetalProgram(
            Device[Device.DEFAULT],
            div_kernel_name,
            compile(
                final_div_kernel.format(
                    local_size=local_size_stage3,
                    global_size=[final_div_gs, 1, 1],
                    n=f"{n}UL",
                )
            ),
        )
        prog_final_div(
            data_out_buf,
            data_in_buf,
            reduction_results_buf,
            global_size=[final_div_gs, 1, 1],
            local_size=local_size_stage3,
        )

    Device[Device.DEFAULT].synchronize()
    custom_time = time.perf_counter() - st_custom

    d0_padded = np.empty((padded_n), dtype=np.float32)
    metalalloc._copyout(flat_mv(d0_padded.data), data_out_buf)
    d0 = d0_padded[:n]

    metalalloc.free(data_in_buf, data_in_buf_size)
    metalalloc.free(data_out_buf, data_out_buf_size)
    metalalloc.free(partial_max_buf, partial_results_buf_size)
    metalalloc.free(partial_sum_buf, partial_results_buf_size)
    metalalloc.free(reduction_results_buf, reduction_results_buf_size)
    return custom_time, d0


def _sched_fusion_5_register(n: int, data: np.ndarray) -> tuple[float, np.ndarray]:
    return _sched_fusion_5_opt_base(
        n,
        data,
        fusion5_reg_reduce_max_kernel,
        fusion5_reg_reduce_sum_kernel,
        final_division,
        "fusion5_reg_reduce_max",
        "fusion5_reg_reduce_sum",
        "final_division",
        is_vectorized=False,
        use_simd=True,
    )


def _sched_fusion_5_vector(n: int, data: np.ndarray) -> tuple[float, np.ndarray]:
    return _sched_fusion_5_opt_base(
        n,
        data,
        fusion5_vec_reduce_max_kernel,
        fusion5_vec_reduce_sum_kernel,
        global_div_opt,
        "fusion5_vec_reduce_max",
        "fusion5_vec_reduce_sum",
        "global_div_opt",
        is_vectorized=True,
        use_simd=False,
    )


def _sched_fusion_3(n: int, data: np.ndarray) -> tuple[float, np.ndarray]:
    local_size_stage1 = [256, 1, 1]
    n = int(n)
    num_workgroup = (n + local_size_stage1[0] - 1) // local_size_stage1[0]
    padded_n = num_workgroup * local_size_stage1[0]

    data_padded = np.full((padded_n), -np.inf, dtype=np.float32)
    data_padded[:n] = data

    data_in_buf_size = padded_n * 4
    data_out_buf_size = padded_n * 4
    intermediate_buf_size = num_workgroup * 2 * 4
    reduction_results_buf_size = 2 * 4
    adjusted_sums_buf_size = num_workgroup * 4

    data_in_buf = metalalloc.alloc(data_in_buf_size)
    metalalloc._copyin(data_in_buf, data_padded.tobytes())
    data_out_buf = metalalloc.alloc(data_out_buf_size)
    intermediate_buf = metalalloc.alloc(intermediate_buf_size)
    reduction_results_buf = metalalloc.alloc(reduction_results_buf_size)
    adjusted_sums_buf = metalalloc.alloc(adjusted_sums_buf_size)

    prog_local_max_and_sum = MetalProgram(
        Device[Device.DEFAULT],
        "local_max_and_sum",
        compile(
            local_max_and_sum.format(
                local_size=local_size_stage1, num_workgroup=num_workgroup
            )
        ),
    )
    prog_final_div = MetalProgram(
        Device[Device.DEFAULT],
        "global_div",
        compile(global_div.format(local_size=local_size_stage1)),
    )
    prog_adjust_sums = MetalProgram(
        Device[Device.DEFAULT],
        "adjust_sums",
        compile(adjust_sums_kernel.format(num_workgroup=num_workgroup)),
    )
    prog_copy_sum = MetalProgram(
        Device[Device.DEFAULT], "copy_sum", compile(copy_sum_kernel)
    )

    st_custom = time.perf_counter()

    prog_local_max_and_sum(
        intermediate_buf,
        data_in_buf,
        global_size=[num_workgroup, 1, 1],
        local_size=local_size_stage1,
    )
    _run_recursive_reduction(
        num_workgroup,
        intermediate_buf,
        "global_reduce_max",
        global_reduce_max,
        result_buf=reduction_results_buf,
    )

    prog_adjust_sums(
        adjusted_sums_buf,
        intermediate_buf,
        reduction_results_buf,
        global_size=[num_workgroup, 1, 1],
        local_size=[min(num_workgroup, 1024), 1, 1],
    )

    final_sum_buf = _run_recursive_reduction(
        num_workgroup, adjusted_sums_buf, "global_sum", global_sum
    )
    prog_copy_sum(
        reduction_results_buf,
        final_sum_buf,
        global_size=[1, 1, 1],
        local_size=[1, 1, 1],
    )
    prog_final_div(
        data_out_buf,
        data_in_buf,
        reduction_results_buf,
        global_size=[num_workgroup, 1, 1],
        local_size=local_size_stage1,
    )

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

    return custom_time, d0


if __name__ == "__main__":
    kernel = {
        "fusion 5": _sched_fusion_5,
        "fusion 5 tuned": None,
        "fusion 5 register": _sched_fusion_5_register,
        "fusion 5 vector": _sched_fusion_5_vector,
        "fusion 3": _sched_fusion_3,
    }
    perf.run(kernel)
