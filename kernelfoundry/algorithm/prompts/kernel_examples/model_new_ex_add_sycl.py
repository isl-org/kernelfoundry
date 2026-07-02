import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom SYCL kernel for element-wise addition
elementwise_add_source = """
#include <sycl/sycl.hpp>
#include <torch/extension.h>
#include <iostream>
#include <c10/xpu/XPUStream.h>

void elementwise_add_sycl_kernel(torch::Tensor a, torch::Tensor b, torch::Tensor out, int size) {
    auto a_data = a.data_ptr<float>();
    auto b_data = b.data_ptr<float>();
    auto out_data = out.data_ptr<float>();

    // Create a SYCL queue
    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();

    // Allocate device memory for a, b, out
    {
        sycl::buffer<float> a_buffer(a_data, sycl::range<1>(size));
        sycl::buffer<float> b_buffer(b_data, sycl::range<1>(size));
        sycl::buffer<float> out_buffer(out_data, sycl::range<1>(size));

        // Submit a command group to the queue
        q.submit([&](sycl::handler& cgh) {
            // Accessors for buffers
            auto a_acc = a_buffer.get_access<sycl::access::mode::read>(cgh);
            auto b_acc = b_buffer.get_access<sycl::access::mode::read>(cgh);
            auto out_acc = out_buffer.get_access<sycl::access::mode::write>(cgh);

            // Kernel for element-wise addition
            cgh.parallel_for<class elementwise_add>(sycl::range<1>(size), [=](sycl::id<1> idx) {
                out_acc[idx] = a_acc[idx] + b_acc[idx];
            });
        });
    }
}

torch::Tensor elementwise_add_sycl(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    // Call the SYCL function
    elementwise_add_sycl_kernel(a, b, out, size);

    return out;
}
"""

elementwise_add_cpp_source = "torch::Tensor elementwise_add_sycl(torch::Tensor a, torch::Tensor b);"

# Compile the inline SYCL code for element-wise addition
elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=elementwise_add_cpp_source,
    sycl_sources=elementwise_add_source,
    functions=["elementwise_add_sycl"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add

    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_sycl(a, b)
