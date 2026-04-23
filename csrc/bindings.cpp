#include <torch/extension.h>

// forward declarations
torch::Tensor cross_entropy_forward(torch::Tensor logits, torch::Tensor labels);
torch::Tensor dot_product_forward(torch::Tensor a, torch::Tensor b);
torch::Tensor prefix_sum_forward(torch::Tensor input);
torch::Tensor matmul_forward(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cross_entropy", &cross_entropy_forward, "Cross entropy loss (CUDA)");
    m.def("dot_product",   &dot_product_forward,   "Dot product (CUDA)");
    m.def("prefix_sum",    &prefix_sum_forward,    "Prefix sum (CUDA)");
    m.def("matmul",        &matmul_forward,         "Matrix multiply (CUDA)");
}