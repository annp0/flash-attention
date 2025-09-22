import triton
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import flash_attn
import pandas as pd
import matplotlib.pyplot as plt

module = load(
    "attn",
    sources =  ["attn.cu", "pybind.cpp"],
    extra_cuda_cflags=["-lineinfo", "--ptxas-options=-v"],
    verbose=True,
)

batch_size = 4
num_heads = 8

len = [1024, 2048, 4096, 8192]

results = []

def run_bench(name, func, l, *args):
    time = triton.testing.do_bench(lambda: func(*args))
    tflops = 4 * batch_size * num_heads * l * l * 128 / time / 1e9
    results.append([name, l, round(time, 2), round(tflops, 2)])


for l in len:
    Q = torch.randn(batch_size, num_heads, l, 128).add(0.4).bfloat16().cuda()
    K = torch.randn(batch_size, num_heads, l, 128).add(0.4).bfloat16().cuda()
    V = torch.randn(batch_size, num_heads, l, 128).add(0.4).bfloat16().cuda()

    out = module.cross_attn(Q, K, V)
    ref = F.scaled_dot_product_attention(Q, K, V)
    torch.testing.assert_close(out, ref, rtol=0.1, atol=0.1)

    run_bench("Torch SPDA", F.scaled_dot_product_attention, l, Q, K, V)
    run_bench("flash-attn", flash_attn.flash_attn_func, l, Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2))
    run_bench("my-attn", module.cross_attn, l, Q, K, V)

df = pd.DataFrame(results, columns=["name", "len", "time (ms)", "TFLOPS"])
print(df)


plt.figure(figsize=(10, 6))
for name in df["name"].unique():
    subset = df[df["name"] == name]
    plt.plot(subset["len"], subset["TFLOPS"], label=name, marker="o")

plt.title("5060 Ti, 2.57GHz, 180W. Time measurement done via cudastream events.")
plt.xlabel("len")
plt.ylabel("TFLOPS")
plt.legend()
plt.grid(True)
plt.savefig("performance_plot.png", dpi=300)
plt.show()