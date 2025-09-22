#include <torch/extension.h>

at::Tensor torch_cross_attn(
    const at::Tensor& Q,
    const at::Tensor& K,
    const at::Tensor& V
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cross_attn", &torch_cross_attn);
}