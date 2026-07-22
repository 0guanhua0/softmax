from functools import cache, partial

import numpy as np
from tinygrad.device import Device
from tinygrad.runtime.ops_metal import (
    MetalAllocator,
    MetalProgram,
    msg,
    objc_instance,
    to_struct,
    wait_check,
)

import perf


@cache
def compile(src: str) -> bytes:
    return Device[Device.DEFAULT].compiler.compile(src)


metalalloc = MetalAllocator(Device[Device.DEFAULT])

k1 = """
#include <metal_stdlib>
using namespace metal;

kernel void k1(device float *data0, device float *data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    size_t gid0 = gid.x;

    float max_val = -FLT_MAX;
    for (size_t i = 0; i < {global_size[0]}; ++i) {{
        max_val = fmax(max_val, data1[i]);
    }}

    float sum_exp = 0.0f;
    for (size_t i = 0; i < {global_size[0]}; ++i) {{
        sum_exp += exp(data1[i] - max_val);
    }}

    data0[gid0] = exp(data1[gid0] - max_val) / sum_exp;
}}
"""

k2 = """
#include <metal_stdlib>
using namespace metal;

kernel void k2(device float *data0, device float *data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    size_t lid0 = lid.x;
    threadgroup float reduce[{local_size[0]}];
    size_t tile = {n_pad} / {local_size[0]};

    float max_val = -FLT_MAX;
    for (size_t i = lid0 * tile; i < (lid0 + 1) * tile; ++i) {{
        max_val = fmax(max_val, data1[i]);
    }}
    reduce[lid0] = max_val;
    for (size_t stride = {local_size[0]} / 2; stride > 0; stride /= 2) {{
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid0 < stride) {{
            reduce[lid0] = fmax(reduce[lid0], reduce[lid0 + stride]);
        }}
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_val = reduce[0];
    float sum_exp = 0.0f;
    for (size_t i = lid0 * tile; i < (lid0 + 1) * tile; ++i) {{
        sum_exp += exp(data1[i] - max_val);
    }}
    reduce[lid0] = sum_exp;
    for (size_t stride = {local_size[0]} / 2; stride > 0; stride /= 2) {{
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid0 < stride) {{
            reduce[lid0] += reduce[lid0 + stride];
        }}
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum_exp = reduce[0];
    for (size_t i = lid0 * tile; i < (lid0 + 1) * tile; ++i) {{
        data0[i] = exp(data1[i] - max_val) / sum_exp;
    }}
}}
"""


k3 = """
#include <metal_stdlib>
using namespace metal;

kernel void k3(device float *data0, device float *data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    size_t lid0 = lid.x;
    threadgroup float reduce[{local_size[0]}];

    float max_val = -FLT_MAX;
    for (size_t i = lid0; i < (size_t){n_pad}; i += {local_size[0]}) {{
        max_val = fmax(max_val, data1[i]);
    }}
    reduce[lid0] = max_val;
    for (size_t stride = {local_size[0]} / 2; stride > 0; stride /= 2) {{
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid0 < stride) {{
            reduce[lid0] = fmax(reduce[lid0], reduce[lid0 + stride]);
        }}
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_val = reduce[0];
    float sum_exp = 0.0f;
    for (size_t i = lid0; i < (size_t){n_pad}; i += {local_size[0]}) {{
        sum_exp += exp(data1[i] - max_val);
    }}
    reduce[lid0] = sum_exp;
    for (size_t stride = {local_size[0]} / 2; stride > 0; stride /= 2) {{
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid0 < stride) {{
            reduce[lid0] += reduce[lid0 + stride];
        }}
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum_exp = reduce[0];
    for (size_t i = lid0; i < (size_t){n_pad}; i += {local_size[0]}) {{
        data0[i] = exp(data1[i] - max_val) / sum_exp;
    }}
}}
"""


k4 = """
#include <metal_stdlib>
using namespace metal;

kernel void k4(device float *data0, device float *data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    size_t lid0 = lid.x;
    size_t simd_id = lid0 / 32;
    threadgroup float reduce[{local_size[0]} / 32];

    float max_val = -FLT_MAX;
    for (size_t i = lid0; i < (size_t){n_pad}; i += {local_size[0]}) {{
        max_val = fmax(max_val, data1[i]);
    }}
    max_val = simd_max(max_val);
    if (lid0 % 32 == 0) {{
        reduce[simd_id] = max_val;
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {{
        max_val = lid0 < {local_size[0]} / 32 ? reduce[lid0] : -FLT_MAX;
        max_val = simd_max(max_val);
        if (lid0 == 0) {{
            reduce[0] = max_val;
        }}
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_val = reduce[0];
    float sum_exp = 0.0f;
    for (size_t i = lid0; i < (size_t){n_pad}; i += {local_size[0]}) {{
        sum_exp += exp(data1[i] - max_val);
    }}
    sum_exp = simd_sum(sum_exp);
    if (lid0 % 32 == 0) {{
        reduce[simd_id] = sum_exp;
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {{
        sum_exp = lid0 < {local_size[0]} / 32 ? reduce[lid0] : 0.0f;
        sum_exp = simd_sum(sum_exp);
        if (lid0 == 0) {{
            reduce[0] = sum_exp;
        }}
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum_exp = reduce[0];
    for (size_t i = lid0; i < (size_t){n_pad}; i += {local_size[0]}) {{
        data0[i] = exp(data1[i] - max_val) / sum_exp;
    }}
}}
"""


k5 = """
#include <metal_stdlib>
using namespace metal;

kernel void k5(device float *data0, device float *data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    size_t lid0 = lid.x;
    size_t simd_id = lid0 / 32;
    threadgroup float reduce_max[{local_size[0]} / 32];
    threadgroup float reduce_sum[{local_size[0]} / 32];

    float max_val = -FLT_MAX;
    float max_val_old = -FLT_MAX;
    float sum_exp = 0.0f;
    for (size_t i = lid0; i < (size_t){n_pad}; i += {local_size[0]}) {{
        max_val = fmax(max_val, data1[i]);
        if (max_val > max_val_old) {{
            sum_exp *= exp(max_val_old - max_val);
            max_val_old = max_val;
        }}
        sum_exp += exp(data1[i] - max_val);
    }}
    max_val = simd_max(max_val);
    sum_exp *= exp(max_val_old - max_val);
    sum_exp = simd_sum(sum_exp);
    if (lid0 % 32 == 0) {{
        reduce_max[simd_id] = max_val;
        reduce_sum[simd_id] = sum_exp;
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {{
        max_val = lid0 < {local_size[0]} / 32 ? reduce_max[lid0] : -FLT_MAX;
        max_val_old = max_val;
        sum_exp = lid0 < {local_size[0]} / 32 ? reduce_sum[lid0] : 0.0f;

        max_val = simd_max(max_val);
        sum_exp *= exp(max_val_old - max_val);
        sum_exp = simd_sum(sum_exp);
        if (lid0 == 0) {{
            reduce_max[0] = max_val;
            reduce_sum[0] = sum_exp;
        }}
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_val = reduce_max[0];
    sum_exp = reduce_sum[0];
    for (size_t i = lid0; i < (size_t){n_pad}; i += {local_size[0]}) {{
        data0[i] = exp(data1[i] - max_val) / sum_exp;
    }}
}}
"""


k6 = """
#include <metal_stdlib>
using namespace metal;

kernel void k6(device float *data0, device float *data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    size_t lid0 = lid.x;
    size_t simd_id = lid0 / 32;
    threadgroup float reduce_max[{local_size[0]} / 32];
    threadgroup float reduce_sum[{local_size[0]} / 32];

    float max_val = -FLT_MAX;
    float max_val_old = -FLT_MAX;
    float sum_exp = 0.0f;
    for (size_t i = lid0; i < (size_t)({n_pad} / 4); i += {local_size[0]}) {{
        float4 val = ((device float4 *)data1)[i];
        max_val = fmax(max_val, val.x);
        max_val = fmax(max_val, val.y);
        max_val = fmax(max_val, val.z);
        max_val = fmax(max_val, val.w);
        if (max_val > max_val_old) {{
            sum_exp *= exp(max_val_old - max_val);
            max_val_old = max_val;
        }}
        sum_exp += exp(val.x - max_val);
        sum_exp += exp(val.y - max_val);
        sum_exp += exp(val.z - max_val);
        sum_exp += exp(val.w - max_val);
    }}
    max_val = simd_max(max_val);
    sum_exp *= exp(max_val_old - max_val);
    sum_exp = simd_sum(sum_exp);
    if (lid0 % 32 == 0) {{
        reduce_max[simd_id] = max_val;
        reduce_sum[simd_id] = sum_exp;
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {{
        max_val = lid0 < {local_size[0]} / 32 ? reduce_max[lid0] : -FLT_MAX;
        max_val_old = max_val;
        sum_exp = lid0 < {local_size[0]} / 32 ? reduce_sum[lid0] : 0.0f;

        max_val = simd_max(max_val);
        sum_exp *= exp(max_val_old - max_val);
        sum_exp = simd_sum(sum_exp);
        if (lid0 == 0) {{
            reduce_max[0] = max_val;
            reduce_sum[0] = sum_exp;
        }}
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_val = reduce_max[0];
    sum_exp = reduce_sum[0];
    for (size_t i = lid0; i < (size_t)({n_pad} / 4); i += {local_size[0]}) {{
        float4 val = ((device float4 *)data1)[i];
        ((device float4 *)data0)[i] = exp(val - max_val) / sum_exp;
    }}
}}
"""


def _sched(
    name: str,
    kernel: str,
    n: int,
    data: np.ndarray,
    global_size: list[int],
    local_size: list[int],
) -> dict:
    pad = local_size[0] * 4
    n_pad = ((n + pad - 1) // pad) * pad
    data0_buf, data1_buf = metalalloc.alloc(n_pad * 4), metalalloc.alloc(n_pad * 4)
    buf = np.frombuffer(metalalloc._as_buffer(data1_buf), dtype=np.float32)
    buf[:n] = data
    if n_pad > n:
        buf[n:] = np.finfo(np.float32).min

    fmt = dict(n_pad=n_pad, global_size=global_size, local_size=local_size)
    prog = MetalProgram(
        Device[Device.DEFAULT],
        name,
        compile(kernel.format(**fmt)),
    )

    def run_fn():
        prog(
            data0_buf,
            data1_buf,
            global_size=global_size,
            local_size=local_size,
            wait=True,
        )

    def copyout_fn():
        return np.frombuffer(metalalloc._as_buffer(data0_buf), dtype=np.float32)[:n]

    def free_fn():
        metalalloc.free(data0_buf, n_pad * 4)
        metalalloc.free(data1_buf, n_pad * 4)
        metalalloc.free_cache()

    return {"run": run_fn, "copyout": copyout_fn, "free": free_fn}


k7_1 = """
#include <metal_stdlib>
using namespace metal;

kernel void k7_1(device float *data0, device float *data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    size_t gid0 = gid.x;
    size_t lid0 = lid.x;

    float max_val0 = -FLT_MAX;
    float max_val1 = -FLT_MAX;
    float max_val2 = -FLT_MAX;
    float max_val3 = -FLT_MAX;
    float sum_exp0 = 0.0f;
    float sum_exp1 = 0.0f;
    float sum_exp2 = 0.0f;
    float sum_exp3 = 0.0f;

    size_t tile = {n_pad} / {global_size[0]};
    size_t addr0 = gid0 * tile / 4;
    for (size_t i = addr0 + lid0; i < addr0 + tile / 4; i += {local_size[0]} * 4) {{
        float4 val0 = ((device float4 *)data1)[i];
        float4 val1 = ((device float4 *)data1)[i + {local_size[0]}];
        float4 val2 = ((device float4 *)data1)[i + {local_size[0]} * 2];
        float4 val3 = ((device float4 *)data1)[i + {local_size[0]} * 3];
        float alu0 = max_val0 > val0.x ? max_val0 : val0.x;
        float alu1 = max_val1 > val1.x ? max_val1 : val1.x;
        float alu2 = max_val2 > val2.x ? max_val2 : val2.x;
        float alu3 = max_val3 > val3.x ? max_val3 : val3.x;
        float alu4 = alu0 > val0.y ? alu0 : val0.y;
        float alu5 = alu1 > val1.y ? alu1 : val1.y;
        float alu6 = alu2 > val2.y ? alu2 : val2.y;
        float alu7 = alu3 > val3.y ? alu3 : val3.y;
        float alu8 = alu4 > val0.z ? alu4 : val0.z;
        float alu9 = alu5 > val1.z ? alu5 : val1.z;
        float alu10 = alu6 > val2.z ? alu6 : val2.z;
        float alu11 = alu7 > val3.z ? alu7 : val3.z;
        float alu12 = alu8 > val0.w ? alu8 : val0.w;
        float alu13 = alu9 > val1.w ? alu9 : val1.w;
        float alu14 = alu10 > val2.w ? alu10 : val2.w;
        float alu15 = alu11 > val3.w ? alu11 : val3.w;
        sum_exp0 *= exp(max_val0 - alu12);
        sum_exp1 *= exp(max_val1 - alu13);
        sum_exp2 *= exp(max_val2 - alu14);
        sum_exp3 *= exp(max_val3 - alu15);
        max_val0 = alu12;
        max_val1 = alu13;
        max_val2 = alu14;
        max_val3 = alu15;
        float alu16 = exp(val0.x - max_val0);
        float alu17 = exp(val1.x - max_val1);
        float alu18 = exp(val2.x - max_val2);
        float alu19 = exp(val3.x - max_val3);
        float alu20 = exp(val0.y - max_val0);
        float alu21 = exp(val1.y - max_val1);
        float alu22 = exp(val2.y - max_val2);
        float alu23 = exp(val3.y - max_val3);
        float alu24 = exp(val0.z - max_val0);
        float alu25 = exp(val1.z - max_val1);
        float alu26 = exp(val2.z - max_val2);
        float alu27 = exp(val3.z - max_val3);
        float alu28 = exp(val0.w - max_val0);
        float alu29 = exp(val1.w - max_val1);
        float alu30 = exp(val2.w - max_val2);
        float alu31 = exp(val3.w - max_val3);
        sum_exp0 += alu16 + alu20 + alu24 + alu28;
        sum_exp1 += alu17 + alu21 + alu25 + alu29;
        sum_exp2 += alu18 + alu22 + alu26 + alu30;
        sum_exp3 += alu19 + alu23 + alu27 + alu31;
    }}
    float alu32 = max_val0 > max_val1 ? max_val0 : max_val1;
    float alu33 = max_val2 > max_val3 ? max_val2 : max_val3;
    float max_val = alu32 > alu33 ? alu32 : alu33;
    sum_exp0 *= exp(max_val0 - max_val);
    sum_exp1 *= exp(max_val1 - max_val);
    sum_exp2 *= exp(max_val2 - max_val);
    sum_exp3 *= exp(max_val3 - max_val);
    float sum_exp = sum_exp0 + sum_exp1 + sum_exp2 + sum_exp3;

    data0[gid0 * {local_size[0]} + lid0] = max_val;
    data0[gid0 * {local_size[0]} + lid0 + {global_size[0]} * {local_size[0]}] = sum_exp;
}}
"""
k7_2 = """
#include <metal_stdlib>
using namespace metal;

kernel void k7_2(device float *data0, device float *data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    size_t lid0 = lid.x;
    size_t simd_id = lid0 / 32;

    threadgroup float reduce_max[{local_size[0]} / 32];
    threadgroup float reduce_sum[{local_size[0]} / 32];

    float thread_max = -FLT_MAX;
    float thread_sum = 0.0f;
    for (size_t i = 0; i < (size_t){global_size[0]}; i++) {{
        float max_val = data1[i * {local_size[0]} + lid0];
        float sum_exp = data1[i * {local_size[0]} + lid0 + {global_size[0]} * {local_size[0]}];
        float new_max = thread_max > max_val ? thread_max : max_val;
        thread_sum = thread_sum * exp(thread_max - new_max) + sum_exp * exp(max_val - new_max);
        thread_max = new_max;
    }}
    float warp_max = simd_max(thread_max);
    float warp_sum = simd_sum(thread_sum * exp(thread_max - warp_max));

    if (lid0 % 32 == 0) {{
        reduce_max[simd_id] = warp_max;
        reduce_sum[simd_id] = warp_sum;
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {{
        thread_max = lid0 < {local_size[0]} / 32 ? reduce_max[lid0] : -FLT_MAX;
        thread_sum = lid0 < {local_size[0]} / 32 ? reduce_sum[lid0] : 0.0f;
        warp_max = simd_max(thread_max);
        warp_sum = simd_sum(thread_sum * exp(thread_max - warp_max));
        if (lid0 == 0) {{
            data0[0] = warp_max;
            data0[1] = 1.0f / warp_sum;
        }}
    }}
}}
"""
k7_3 = """
#include <metal_stdlib>
using namespace metal;

kernel void k7_3(device float *data0, device float *data1, device float *data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    size_t gid0 = gid.x;
    size_t lid0 = lid.x;

    float max_val = data2[0];
    float sum_exp_inv = data2[1];
    size_t tile = {n_pad} / {global_size[0]};
    size_t addr0 = gid0 * tile / 4;
    for (size_t i = addr0 + lid0; i < addr0 + tile / 4; i += {local_size[0]} * 4) {{
        float4 val0 = ((device float4 *)data1)[i];
        float4 val1 = ((device float4 *)data1)[i + {local_size[0]}];
        float4 val2 = ((device float4 *)data1)[i + {local_size[0]} * 2];
        float4 val3 = ((device float4 *)data1)[i + {local_size[0]} * 3];
        ((device float4 *)data0)[i] = exp(val0 - max_val) * sum_exp_inv;
        ((device float4 *)data0)[i + {local_size[0]}] = exp(val1 - max_val) * sum_exp_inv;
        ((device float4 *)data0)[i + {local_size[0]} * 2] = exp(val2 - max_val) * sum_exp_inv;
        ((device float4 *)data0)[i + {local_size[0]} * 3] = exp(val3 - max_val) * sum_exp_inv;
    }}
}}
"""


def _sched_7(
    n: int,
    data: np.ndarray,
    global_size: list[int],
    local_size: list[int],
) -> dict:
    pad = global_size[0] * local_size[0] * 16
    n_pad = ((n + pad - 1) // pad) * pad

    data0_buf = metalalloc.alloc(n_pad * 4)
    data1_buf = metalalloc.alloc(n_pad * 4)
    data2_buf = metalalloc.alloc(global_size[0] * local_size[0] * 2 * 4)
    data_glob = metalalloc.alloc(2 * 4)

    buf = np.frombuffer(metalalloc._as_buffer(data1_buf), dtype=np.float32)
    buf[:n] = data
    if n_pad > n:
        buf[n:] = np.finfo(np.float32).min

    fmt = dict(n_pad=n_pad, global_size=global_size, local_size=local_size)
    p1 = MetalProgram(Device[Device.DEFAULT], "k7_1", compile(k7_1.format(**fmt)))
    p2 = MetalProgram(Device[Device.DEFAULT], "k7_2", compile(k7_2.format(**fmt)))
    p3 = MetalProgram(Device[Device.DEFAULT], "k7_3", compile(k7_3.format(**fmt)))

    dev = Device[Device.DEFAULT]

    def run_fn():
        command_buffer = msg("commandBuffer", objc_instance)(dev.mtl_queue)

        enc = msg("computeCommandEncoder", objc_instance)(command_buffer)
        msg("setComputePipelineState:")(enc, p1.pipeline_state)
        msg("setBuffer:offset:atIndex:")(enc, data2_buf.buf, data2_buf.offset, 0)
        msg("setBuffer:offset:atIndex:")(enc, data1_buf.buf, data1_buf.offset, 1)
        msg("dispatchThreadgroups:threadsPerThreadgroup:")(
            enc, to_struct(*global_size), to_struct(*local_size)
        )
        msg("endEncoding")(enc)

        enc = msg("computeCommandEncoder", objc_instance)(command_buffer)
        msg("setComputePipelineState:")(enc, p2.pipeline_state)
        msg("setBuffer:offset:atIndex:")(enc, data_glob.buf, data_glob.offset, 0)
        msg("setBuffer:offset:atIndex:")(enc, data2_buf.buf, data2_buf.offset, 1)
        msg("dispatchThreadgroups:threadsPerThreadgroup:")(
            enc, to_struct(1, 1, 1), to_struct(*local_size)
        )
        msg("endEncoding")(enc)

        enc = msg("computeCommandEncoder", objc_instance)(command_buffer)
        msg("setComputePipelineState:")(enc, p3.pipeline_state)
        msg("setBuffer:offset:atIndex:")(enc, data0_buf.buf, data0_buf.offset, 0)
        msg("setBuffer:offset:atIndex:")(enc, data1_buf.buf, data1_buf.offset, 1)
        msg("setBuffer:offset:atIndex:")(enc, data_glob.buf, data_glob.offset, 2)
        msg("dispatchThreadgroups:threadsPerThreadgroup:")(
            enc, to_struct(*global_size), to_struct(*local_size)
        )
        msg("endEncoding")(enc)

        msg("commit")(command_buffer)
        wait_check(command_buffer)

    def copyout_fn():
        return np.frombuffer(metalalloc._as_buffer(data0_buf), dtype=np.float32)[:n]

    def free_fn():
        metalalloc.free(data0_buf, n_pad * 4)
        metalalloc.free(data1_buf, n_pad * 4)
        metalalloc.free(data2_buf, global_size[0] * local_size[0] * 2 * 4)
        metalalloc.free(data_glob, 2 * 4)
        metalalloc.free_cache()

    return {"run": run_fn, "copyout": copyout_fn, "free": free_fn}


if __name__ == "__main__":
    kernel = {
        "2": partial(_sched, "k2", k2),
        "3": partial(_sched, "k3", k3),
        "4": partial(_sched, "k4", k4),
        "5": partial(_sched, "k5", k5),
        "6": partial(_sched, "k6", k6),
        "7": _sched_7,
    }
    perf.run(kernel)
