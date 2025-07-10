# [FastSoftmax](https://github.com/SzymonOzog/FastSoftmax) on [tinygrad](https://github.com/tinygrad/tinygrad)

$$\Large\text{softmax}(x_i) = \frac{e^{x_i - max(x)}}{\sum_{j=1}^{K} e^{x_j - max(x)}}$$

# background
[How DRAM works and why should you care | GPU Programming](https://www.youtube.com/watch?v=huhg3V4ZRW0)

opertaion:
- memory-bound reduction
- compute-bound element-wise operation

thread = basic unit

warp = thread group

occupancy = active warp / max warp 

resource light kernel = high occupancy

high occupancy can hide memory latency. when one thread group wait for data transfer, other group can run

# kernel
## base
one kernel

## fusion 5
break down to small kernel:
- reduce local max
- reduce global max
- reduce local exp
- reduce global exp
- div

## fusion 5 tuned
tune param for each kernel

## fusion 5 fast exp
[Approximation of The Power Function](https://gudgud96.github.io/2024/01/02/exp-2/)

$$e^x = 2^{x \log_2 e}$$
$$x' = x \log_2 e \quad \implies \quad 2^{x'} = 2^{i+f} = 2^i \cdot 2^f$$

IEEE-754 32 bit floating point representation:

$$\text{bits}(2^i) = (i + 127) << 23$$

$$ 2^f \approx 0.0570f^3 + 0.2486f^2 + 0.6928f + 0.9992 $$

## fusion 5 register
share data at register level

## fusion 5 vector
load data in vector. less instruction

## fusion 3
exp(x - max) = exp(x - local\_max) * exp(local\_max - global\_max)

- reduce local max and sum
- reduce global max and sum
- div

small input: fusion 3 save kernel launch overhead and memory pass

big input: fusion 5 simple, high-occupancy kernels are better at hiding memory latency, leading to higher effective memory bandwidth and performance.

# benchmark
![benchmark](benchmark.png)

# tip
kiss: keep it simple stupid
