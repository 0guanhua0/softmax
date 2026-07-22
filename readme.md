# [FastSoftmax](https://github.com/SzymonOzog/FastSoftmax) on [tinygrad](https://github.com/tinygrad/tinygrad)

$$\Large\text{softmax}(x_i) = \frac{e^{x_i - max(x)}}{\sum_{j=1}^{K} e^{x_j - max(x)}}$$

[How DRAM works and why should you care | GPU Programming](https://www.youtube.com/watch?v=huhg3V4ZRW0)

# kernel
data = [d1, d2, d3, d4]

## 1
thread 1 max(d1, d2, d3, d4) -> sum(exp(data[i] - max)) -> exp(d1 - max) / sum
thread 2 max(d1, d2, d3, d4) -> sum(exp(data[i] - max)) -> exp(d2 - max) / sum
thread 3 max(d1, d2, d3, d4) -> sum(exp(data[i] - max)) -> exp(d3 - max) / sum
thread 4 max(d1, d2, d3, d4) -> sum(exp(data[i] - max)) -> exp(d4 - max) / sum

# perf
![perf](perf.png)
