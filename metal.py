from functools import cache, partial

import numpy as np
from tinygrad.device import Device
from tinygrad.helpers import flat_mv
from tinygrad.runtime.ops_metal import MetalAllocator, MetalProgram

import perf


@cache
def compile(src: str) -> bytes:
    return Device[Device.DEFAULT].compiler.compile(src)


metalalloc = MetalAllocator(Device[Device.DEFAULT])

k1 = """
#include <metal_stdlib>
using namespace metal;

kernel void k1(device float *d0, device float *d1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    size_t gid0 = gid.x;

    float max_val = -FLT_MAX;
    for (size_t i = 0; i < {global_size[0]}; ++i) {{
        max_val = fmax(max_val, d1[i]);
    }}

    float sum_exp = 0.0f;
    for (size_t i = 0; i < {global_size[0]}; ++i) {{
        sum_exp += exp(d1[i] - max_val);
    }}

    d0[gid0] = exp(d1[gid0] - max_val) / sum_exp;
}}
"""

k2 = """
#include <metal_stdlib>
using namespace metal;

kernel void k2(device float *d0, device float *d1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    size_t lid0 = lid.x;
    threadgroup float reduce[{local_size[0]}];
    size_t BLOCK = ({n} + {local_size[0]} - 1) / {local_size[0]};

    float max_val = -FLT_MAX;
    for (size_t i = lid0 * BLOCK; i < min((lid0 + 1) * BLOCK, (size_t){n}); ++i) {{
        max_val = fmax(max_val, d1[i]);
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
    for (size_t i = lid0 * BLOCK; i < min((lid0 + 1) * BLOCK, (size_t){n}); ++i) {{
        sum_exp += exp(d1[i] - max_val);
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
    for (size_t i = lid0 * BLOCK; i < min((lid0 + 1) * BLOCK, (size_t){n}); ++i) {{
        d0[i] = exp(d1[i] - max_val) / sum_exp;
    }}
}}
"""


k3 = """
#include <metal_stdlib>
using namespace metal;

kernel void k3(device float *d0, device float *d1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    size_t lid0 = lid.x;
    threadgroup float reduce[{local_size[0]}];

    float max_val = -FLT_MAX;
    for (size_t i = lid0; i < (size_t){n}; i += {local_size[0]}) {{
        max_val = fmax(max_val, d1[i]);
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
    for (size_t i = lid0; i < (size_t){n}; i += {local_size[0]}) {{
        sum_exp += exp(d1[i] - max_val);
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
    for (size_t i = lid0; i < (size_t){n}; i += {local_size[0]}) {{
        d0[i] = exp(d1[i] - max_val) / sum_exp;
    }}
}}
"""


k4 = """
#include <metal_stdlib>
using namespace metal;

kernel void k4(device float *d0, device float *d1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
    size_t lid0 = lid.x;
    size_t simd_id = lid0 / 32;
    threadgroup float reduce[{local_size[0]} / 32];

    float max_val = -FLT_MAX;
    for (size_t i = lid0; i < (size_t){n}; i += {local_size[0]}) {{
        max_val = fmax(max_val, d1[i]);
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
    for (size_t i = lid0; i < (size_t){n}; i += {local_size[0]}) {{
        sum_exp += exp(d1[i] - max_val);
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
    for (size_t i = lid0; i < (size_t){n}; i += {local_size[0]}) {{
        d0[i] = exp(d1[i] - max_val) / sum_exp;
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
) -> np.ndarray:
    prog = MetalProgram(
        Device[Device.DEFAULT],
        name,
        compile(kernel.format(n=n, global_size=global_size, local_size=local_size)),
    )

    d0_buf, d1_buf = metalalloc.alloc(n * 4), metalalloc.alloc(n * 4)
    metalalloc._copyin(d1_buf, data.tobytes())

    prog(d0_buf, d1_buf, global_size=global_size, local_size=local_size, wait=True)

    d0 = np.empty((n), dtype=np.float32)
    metalalloc._copyout(flat_mv(d0.data), d0_buf)

    metalalloc.free(d0_buf, n * 4)
    metalalloc.free(d1_buf, n * 4)

    return d0


if __name__ == "__main__":
    kernel = {
        "2": partial(_sched, "k2", k2),
        "3": partial(_sched, "k3", k3),
        "4": partial(_sched, "k4", k4),
    }
    perf.run(kernel)
