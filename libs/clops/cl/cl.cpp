#include "common.hpp"

#include <cassert>
#include <iostream>
#include <map>
#include <variant>
#include <sstream>

#ifdef __linux__
#    include <dlfcn.h>
#endif
#include <omp.h>

#ifndef ASSERT
#    define ASSERT(cond)                                                     \
        if (!(cond)) {                                                       \
            std::stringstream ss;                                            \
            ss << __FILE__ << ":" << __LINE__ << " " << #cond << " failed!"; \
            throw std::runtime_error(ss.str());                              \
        }
#endif

//=======================================================================================================
#define ANSI_COLOR_INFO  "\033[32m"
#define ANSI_COLOR_ERROR "\033[31m"
#define ANSI_COLOR_RESET "\033[0m"

// CPU side generates GPU tasks:
//  - shape-infer
//  - allocate buffer from pool
//  - choose suitable (pre-built) kernel and enqueue it with parameters
//  - free unused buffer back to pool（by using smart-pointer/life-scope)
//
// and CPU can do this w/o waiting for GPU to finish, so CPU is like a JIT compiler spills GPU tasks on the fly.
// so CPU logic is simple and we can use python for that purpose w/o worry about performance
//
// CPU can also break anytime on any GPU tensor by waiting queue to finish, and copy data back to CPU side numpy array to check results
//
// because CPU does shape-infer, we need combine shape-information with cl::Buffer to form a clTensor python object
// and this object:
//    - is created from pool
//    - with shape/strides information
//    - can fill with numpy array data with compatible shape
//    - can copy from GPU back to CPU side
//    - can be passed to a OpenCL kernel
//
// to keep things simpler, we don't use out-of-order queue so no event concept is used too.
// (just relying on out-side profiling tools to optimize)
//
static auto CLDEBUG = std::getenv("CLDEBUG") ? atoi(std::getenv("CLDEBUG")) : 0;

#define DEBUG_MEMPOOL_SUMMARY (CLDEBUG & 1)
#define DEBUG_MEMPOOL_VERBOSE (CLDEBUG & 2)
#define DEBUG_MEMPOOL_HOST (CLDEBUG & 4)

// Constructed lazily on first use (inside a pybind11-bound call, not at module-load
// time) so `import clops` doesn't require a GPU/OpenCL device to be present.
static ocl_queue& g_queue_instance() {
    static ocl_queue instance;
    return instance;
}
#define g_queue g_queue_instance()

// all tensors are device memory managed by buffer_pool
struct buffer_pool {
    std::multimap<size_t, void*> pool;
    size_t total_size = 0;
    size_t total_count = 0;
    size_t total_alloc_size = 0;
    size_t total_alloc_count = 0;

    void* alloc(size_t sz) {
        total_size += sz;
        total_count++;

        auto it = pool.find(sz);
        if (it != pool.end()) {
            void* p = it->second;
            pool.erase(it);
            if (DEBUG_MEMPOOL_VERBOSE)
                std::cout << "[buffer_pool] alloc from pool " << sz << " bytes @ " << p << std::endl;
            return p;
        }
        // allocate a new one
        total_alloc_size += sz;
        total_alloc_count++;

        void* p;
        if (DEBUG_MEMPOOL_HOST)
            p = g_queue.malloc_host(sz);
        else
            p = g_queue.malloc_device(sz);
        if (DEBUG_MEMPOOL_VERBOSE)
            std::cout << "[buffer_pool] alloc new " << sz << " bytes @ " << p << std::endl;
        return p;
    }

    void free(void* buff, size_t sz) {
        pool.emplace(sz, buff);
        if (DEBUG_MEMPOOL_VERBOSE)
            std::cout << "[buffer_pool] free " << sz << " bytes @ " << buff << std::endl;
    }

    void show() {
        std::cout << "=== buffer_pool(using '" << (DEBUG_MEMPOOL_HOST ? "malloc_host" : "malloc_device") << "') ===" << std::endl;
        size_t pool_size = 0;
        std::map<size_t, int> summary;
        for (auto const& p : pool) {
            auto sz = p.first;
            if (summary.count(sz) == 0)
                summary[sz] = 0;
            summary[sz]++;
            pool_size += p.first;
        }
        for (auto const& p : summary) {
            std::cout << "\t " << p.first << " bytes x " << p.second << std::endl;
        }
        std::cout << "=== totally pool/total/actual:  " << pool.size() << "/" << total_count << "/" << total_alloc_count << " buffers"
                  << " pool/total/actual = " << pool_size / 1e6 << "/" << total_size / 1e6 << "/" << total_alloc_size / 1e6 << " MB  ===" << std::endl;
    }

    ~buffer_pool() {
        if (DEBUG_MEMPOOL_SUMMARY)
            show();
    }
};
static buffer_pool g_buff_pool;

void tensor::resize(const std::vector<cl_uint>& dims, py::dtype dtype) {
    dt = dtype;
    auto it_dims = dims.begin();
    auto it_dims_end = dims.end();
    numel = 1;
    for (; it_dims != it_dims_end; ++it_dims) {
        auto dim = *it_dims;
        shape.push_back(dim);
        numel *= dim;
    }

    cl_uint stride = 1;
    strides.resize(shape.size());
    for (int i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    update_buff();
}

void tensor::update_buff() {
    size_t sz = numel * dt.itemsize();
    void* p = g_buff_pool.alloc(sz);
    p_buff = std::shared_ptr<void>(p, [sz](void* pbuff) {
        g_buff_pool.free(pbuff, sz);
    });
}

void tensor::resize(const py::array& b) {
    py::buffer_info info = b.request();
    dt = b.dtype();
    cl_uint expect_stride = 1;
    numel = 1;
    shape.resize(info.ndim);
    strides.resize(info.ndim);
    for (int i = info.ndim - 1; i >= 0; --i) {
        numel *= info.shape[i];
        shape[i] = info.shape[i];
        strides[i] = info.strides[i] / info.itemsize;
        ASSERT(strides[i] == expect_stride);
        expect_stride *= shape[i];
    }
    update_buff();
    auto* p_host = reinterpret_cast<uint8_t*>(info.ptr);
    g_queue.memcpy_HtoD(p_buff.get(), p_host, numel * dt.itemsize());
}

py::array tensor::to_numpy() {
    // this shouldn't be a very frequent operation which requires optimizations
    // so we just allocate
    if (!p_buff) return {};

    py::array ret(dt, shape);
    py::buffer_info info = ret.request();
    auto* p_host = reinterpret_cast<uint8_t*>(info.ptr);

    g_queue.memcpy_DtoH(p_host, p_buff.get(), numel * dt.itemsize());
    return ret;
}

//======================================================================================================
// https://www.iwocl.org/wp-content/uploads/39-presentation-iwocl-syclcon-2022-aksel.pdf
static const char* get_ocl_error_string(cl_int error) {
#define CASE_CL_ERROR(x) \
    case x:              \
        return #x;
    switch (error) {
        // run-time and JIT compiler errors
        CASE_CL_ERROR(CL_SUCCESS);
        CASE_CL_ERROR(CL_DEVICE_NOT_FOUND);
        CASE_CL_ERROR(CL_DEVICE_NOT_AVAILABLE);
        CASE_CL_ERROR(CL_COMPILER_NOT_AVAILABLE);
        CASE_CL_ERROR(CL_MEM_OBJECT_ALLOCATION_FAILURE);
        CASE_CL_ERROR(CL_OUT_OF_RESOURCES);
        CASE_CL_ERROR(CL_OUT_OF_HOST_MEMORY);
        CASE_CL_ERROR(CL_PROFILING_INFO_NOT_AVAILABLE);
        CASE_CL_ERROR(CL_MEM_COPY_OVERLAP);
        CASE_CL_ERROR(CL_IMAGE_FORMAT_MISMATCH);
        CASE_CL_ERROR(CL_IMAGE_FORMAT_NOT_SUPPORTED);
        CASE_CL_ERROR(CL_BUILD_PROGRAM_FAILURE);
        CASE_CL_ERROR(CL_MAP_FAILURE);
        CASE_CL_ERROR(CL_MISALIGNED_SUB_BUFFER_OFFSET);
        CASE_CL_ERROR(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
        CASE_CL_ERROR(CL_COMPILE_PROGRAM_FAILURE);
        CASE_CL_ERROR(CL_LINKER_NOT_AVAILABLE);
        CASE_CL_ERROR(CL_LINK_PROGRAM_FAILURE);
        CASE_CL_ERROR(CL_DEVICE_PARTITION_FAILED);
        CASE_CL_ERROR(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
        CASE_CL_ERROR(CL_INVALID_VALUE);
        CASE_CL_ERROR(CL_INVALID_DEVICE_TYPE);
        CASE_CL_ERROR(CL_INVALID_PLATFORM);
        CASE_CL_ERROR(CL_INVALID_DEVICE);
        CASE_CL_ERROR(CL_INVALID_CONTEXT);
        CASE_CL_ERROR(CL_INVALID_QUEUE_PROPERTIES);
        CASE_CL_ERROR(CL_INVALID_COMMAND_QUEUE);
        CASE_CL_ERROR(CL_INVALID_HOST_PTR);
        CASE_CL_ERROR(CL_INVALID_MEM_OBJECT);
        CASE_CL_ERROR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
        CASE_CL_ERROR(CL_INVALID_IMAGE_SIZE);
        CASE_CL_ERROR(CL_INVALID_SAMPLER);
        CASE_CL_ERROR(CL_INVALID_BINARY);
        CASE_CL_ERROR(CL_INVALID_BUILD_OPTIONS);
        CASE_CL_ERROR(CL_INVALID_PROGRAM);
        CASE_CL_ERROR(CL_INVALID_PROGRAM_EXECUTABLE);
        CASE_CL_ERROR(CL_INVALID_KERNEL_NAME);
        CASE_CL_ERROR(CL_INVALID_KERNEL_DEFINITION);
        CASE_CL_ERROR(CL_INVALID_KERNEL);
        CASE_CL_ERROR(CL_INVALID_ARG_INDEX);
        CASE_CL_ERROR(CL_INVALID_ARG_VALUE);
        CASE_CL_ERROR(CL_INVALID_ARG_SIZE);
        CASE_CL_ERROR(CL_INVALID_KERNEL_ARGS);
        CASE_CL_ERROR(CL_INVALID_WORK_DIMENSION);
        CASE_CL_ERROR(CL_INVALID_WORK_GROUP_SIZE);
        CASE_CL_ERROR(CL_INVALID_WORK_ITEM_SIZE);
        CASE_CL_ERROR(CL_INVALID_GLOBAL_OFFSET);
        CASE_CL_ERROR(CL_INVALID_EVENT_WAIT_LIST);
        CASE_CL_ERROR(CL_INVALID_EVENT);
        CASE_CL_ERROR(CL_INVALID_OPERATION);
        CASE_CL_ERROR(CL_INVALID_GL_OBJECT);
        CASE_CL_ERROR(CL_INVALID_BUFFER_SIZE);
        CASE_CL_ERROR(CL_INVALID_MIP_LEVEL);
        CASE_CL_ERROR(CL_INVALID_GLOBAL_WORK_SIZE);
        CASE_CL_ERROR(CL_INVALID_PROPERTY);
        CASE_CL_ERROR(CL_INVALID_IMAGE_DESCRIPTOR);
        CASE_CL_ERROR(CL_INVALID_COMPILER_OPTIONS);
        CASE_CL_ERROR(CL_INVALID_LINKER_OPTIONS);
        CASE_CL_ERROR(CL_INVALID_DEVICE_PARTITION_COUNT);
        CASE_CL_ERROR(CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR);
        CASE_CL_ERROR(CL_PLATFORM_NOT_FOUND_KHR);
    // CASE_CL_ERROR(CL_INVALID_D3D10_DEVICE_KHR);
    // CASE_CL_ERROR(CL_INVALID_D3D10_RESOURCE_KHR);
    // CASE_CL_ERROR(CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR);
    // CASE_CL_ERROR(CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR);
    default:
        return "Unknown OpenCL error";
    }
}

void cl_kernels::check_error(const char* name, cl_int error, bool show_build_log) {
    if (error != CL_SUCCESS) {
        std::stringstream ss;
        ss << name << " failed with " << get_ocl_error_string(error) << " (" << error << ")";

        if (show_build_log) {
            size_t len = 0;
            cl_int ret = CL_SUCCESS;
            ret = clGetProgramBuildInfo(m_prog, g_queue.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
            if (ret != CL_SUCCESS) {
                ss << "clGetProgramBuildInfo(..., CL_PROGRAM_BUILD_LOG, ...) failed with " << get_ocl_error_string(ret) << " (" << ret << ")";
                throw std::runtime_error(ss.str());
            }
            std::string log;
            log.resize(len + 1, '\0');
            ret = clGetProgramBuildInfo(m_prog, g_queue.device, CL_PROGRAM_BUILD_LOG, len, log.data(), NULL);
            if (ret != CL_SUCCESS) {
                ss << "clGetProgramBuildInfo(..., CL_PROGRAM_BUILD_LOG, ...) failed with " << get_ocl_error_string(ret) << " (" << ret << ")";
                throw std::runtime_error(ss.str());
            }

            // ss << "build failed on device: " << sycl_queue.get_device().get_info<sycl::info::device::name>() << std::endl;
            ss << "build options: " << m_options << std::endl;

            ss << ANSI_COLOR_ERROR << log << ANSI_COLOR_RESET << std::endl;
        }
        throw std::runtime_error(ss.str());
    }
}

cl_kernels::cl_kernels(py::bytes bin_bytes, std::string options) : m_options(options) {
    std::string content{bin_bytes};
    size_t lengths[1] = {content.size()};
    const unsigned char* binaries[1] = { reinterpret_cast<unsigned char*>(content.data()) };
    cl_int binary_status[1] = {0};
    cl_int errcodes[1] = {0};
    m_prog = clCreateProgramWithBinary(g_queue.context, 1, &g_queue.device, lengths, binaries, binary_status, errcodes);
    check_error("clCreateProgramWithBinary", errcodes[0]);

    cl_int build_error = ::clBuildProgram(m_prog, 1, &g_queue.device, m_options.c_str(), nullptr, nullptr);
    check_error("clBuildProgram", build_error, true);
}

void cl_kernels::setup(std::string source, std::string options, std::string dump_dir) {
    m_source = source;
    m_options = options;

    // C for metal suppport
    if (options.find("-cmc") == std::string::npos)
        m_options = options + " -cl-std=CL3.0";

    cl_int build_error = CL_SUCCESS;
    const char* strings = source.c_str();
    const size_t length = source.size();
    m_prog = ::clCreateProgramWithSource(g_queue.context, (cl_uint)1, &strings, &length, &build_error);
    check_error("clCreateProgramWithSource", build_error);
    build_error = ::clBuildProgram(m_prog, 1, &g_queue.device, m_options.c_str(), nullptr, nullptr);
    check_error("clBuildProgram", build_error, true);
}

cl_kernels::cl_kernels(std::string source, std::string options, std::string dump_dir) {
    setup(source, options, dump_dir);
}

cl_kernel cl_kernels::get_kernel(std::string kernel_name) {
    cl_int err = CL_SUCCESS;
    auto it = kernel_map.find(kernel_name);
    if (it == kernel_map.end()) {
        cl_kernel kernel = clCreateKernel(m_prog, kernel_name.c_str(), &err);
        if (err != CL_SUCCESS) {
            std::stringstream ss;
            ss << "clCreateKernel(\"" << kernel_name << "\", ...) failed with " << get_ocl_error_string(err) << " (" << err << ")";
            throw std::runtime_error(ss.str());
        }
        kernel_map[kernel_name] = kernel;
    }
    return kernel_map[kernel_name];
}

void cl_kernels::set_arg(cl_kernel kernel, int idx, int arg) {
    ::clSetKernelArg(kernel, idx, sizeof(arg), &arg);
}
void cl_kernels::set_arg(cl_kernel kernel, int idx, float arg) {
    ::clSetKernelArg(kernel, idx, sizeof(arg), &arg);
}
void cl_kernels::set_arg(cl_kernel kernel, int idx, void* arg) {
    ::clSetKernelArgSVMPointer(kernel, idx, arg);
}

void cl_kernels::enqueue(cl_kernel kernel, const std::vector<size_t>& global_size, const std::vector<size_t>& local_size) {
    cl_int err = CL_SUCCESS;

    cl_event event;
    err = ::clEnqueueNDRangeKernel(g_queue.queue, kernel, global_size.size(), nullptr, global_size.data(), local_size.data(), 0, nullptr, &event);
    if (err != CL_SUCCESS) {
        std::stringstream ss;
        ss << "clEnqueueNDRangeKernel(...) failed with " << get_ocl_error_string(err) << " (" << err << ")";
        throw std::runtime_error(ss.str());
    }
    g_queue.events.emplace_back(event);
    return;
}

#define CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL      0x4200
#define CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL    0x4201
#define CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL    0x4202

int cl_kernels::SetKernelExecInfo(std::string kernel_name, int i, bool enable) {
    auto kernel = get_kernel(kernel_name);
    cl_uint param = enable ? CL_TRUE : CL_FALSE;
    return clSetKernelExecInfo(
    //return g_queue.call_ext<cl_int>("clSetKernelExecInfo",
                             kernel,
                             CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL + i,
                             sizeof(param),
                             &param);
}

void cl_kernels::enqueue_py(std::string kernel_name, const std::vector<size_t>& global_size, const std::vector<size_t>& local_size, py::args args) {
    cl_int err = CL_SUCCESS;
    auto kernel = get_kernel(kernel_name);

    cl_uint nargs;
    err = ::clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(nargs), &nargs, nullptr);
    if (err != CL_SUCCESS) {
        std::stringstream ss;
        ss << "clGetKernelInfo(\"" << kernel_name << "\", CL_KERNEL_NUM_ARGS, ...) failed with " << get_ocl_error_string(err) << " (" << err << ")";
        throw std::runtime_error(ss.str());
    }
    int arg_idx = 0;
    for (auto& arg : args) {
        if (arg_idx >= nargs)
            throw std::runtime_error(std::string("arg index ") + std::to_string(arg_idx) + " exceeds nargs=" + std::to_string(nargs));
        if (py::isinstance<py::int_>(arg)) {
            auto i = arg.cast<int>();
            ::clSetKernelArg(kernel, arg_idx, sizeof(i), &i);
        } else if (py::isinstance<py::float_>(arg)) {
            auto f = arg.cast<float>();
            ::clSetKernelArg(kernel, arg_idx, sizeof(f), &f);
        } else if (py::isinstance<tensor>(arg)) {
            const auto& t = arg.cast<tensor>();
            //::clSetKernelArgSVMPointer(kernel, arg_idx, static_cast<void*>(t));
            using clSetKernelArgMemPointerINTEL_func_t = cl_int (*)(cl_kernel, cl_uint, const void *);
            static ext_func_t<clSetKernelArgMemPointerINTEL_func_t> ext_func(
                "clSetKernelArgMemPointerINTEL");
            ext_func(g_queue.platform, kernel, arg_idx, static_cast<void*>(t));
        } else if (arg.is_none()) {
            ::clSetKernelArgSVMPointer(kernel, arg_idx, static_cast<void*>(nullptr));
        } else {
            throw std::runtime_error(std::string("Unknown kernel arg at index ") + std::to_string(arg_idx));
        }
        arg_idx++;
    }
    if (arg_idx < nargs)
        throw std::runtime_error(std::string("arg count ") + std::to_string(arg_idx) + " smaller than expected nargs=" + std::to_string(nargs));

    cl_event event;
    err = ::clEnqueueNDRangeKernel(g_queue.queue, kernel, global_size.size(), nullptr, global_size.data(), local_size.data(), 0, nullptr, &event);
    if (err != CL_SUCCESS) {
        std::stringstream ss;
        ss << "clEnqueueNDRangeKernel(\"" << kernel_name << "\",...) failed with " << get_ocl_error_string(err) << " (" << err << ")";
        throw std::runtime_error(ss.str());
    }
    g_queue.events.emplace_back(event);
    return;
}

std::string cl_kernels::info(const char* kernel_name, const std::vector<int>& local_size, size_t sub_groups) {
    return "";
}

#if 0
std::string cl_kernels::info(const char* kernel_name, const std::vector<int>& local_size, size_t sub_groups) {
    std::stringstream ss;
    cl::NDRange local_work_size = to_ndrange(local_size);
    auto device = cl::Device::getDefault();
    auto& k = kernel_map[kernel_name];

    ss << kernel_name << " [getWorkGroupInfo] :" << "\n";
    ss << "    CL_KERNEL_WORK_GROUP_SIZE: " << k.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << "\n";
    ss << "    CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: " << k.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device) << "\n";
    // std::cout << "    CL_KERNEL_GLOBAL_WORK_SIZE: " << k.getWorkGroupInfo<CL_KERNEL_GLOBAL_WORK_SIZE>(device) << "\n";
    ss << "    CL_KERNEL_COMPILE_WORK_GROUP_SIZE: " << k.getWorkGroupInfo<CL_KERNEL_COMPILE_WORK_GROUP_SIZE>(device) << "\n";
    ss << "    CL_KERNEL_LOCAL_MEM_SIZE: " << k.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(device) << "\n";
    ss << "    CL_KERNEL_PRIVATE_MEM_SIZE: " << k.getWorkGroupInfo<CL_KERNEL_PRIVATE_MEM_SIZE>(device) << "\n";
    ss << "    CL_KERNEL_SPILL_MEM_SIZE_INTEL: " << k.getWorkGroupInfo<CL_KERNEL_SPILL_MEM_SIZE_INTEL>(device) << "\n";

    ss << kernel_name << " [getSubGroupInfo] :" << "\n";
    ss << "    CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE: " << local_work_size << " is " << k.getSubGroupInfo<CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE>(device, local_work_size)
        << "\n";
    ss << "    CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE: " << local_work_size << " is " << k.getSubGroupInfo<CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE>(device, local_work_size)
        << "\n";
    ss << "    CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT: " << sub_groups << " is " << k.getSubGroupInfo<CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT>(device, sub_groups) << "\n";
    ss << "    CL_KERNEL_MAX_NUM_SUB_GROUPS: " << k.getSubGroupInfo<CL_KERNEL_MAX_NUM_SUB_GROUPS>(device, local_work_size) << "\n";
    ss << "    CL_KERNEL_COMPILE_NUM_SUB_GROUPS: " << k.getSubGroupInfo<CL_KERNEL_COMPILE_NUM_SUB_GROUPS>(device, local_work_size) << "\n";

    auto nargs = k.getInfo<CL_KERNEL_NUM_ARGS>();
    ss << " args " << nargs << " :" << std::endl;
    for (int arg_idx = 0; arg_idx < nargs; arg_idx++) {
        ss << "\t" << arg_idx << " " << k.getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(arg_idx) << " " << k.getArgInfo<CL_KERNEL_ARG_NAME>(arg_idx) << std::endl;
    }
    return ss.str();
}
#endif

static bool enable_profile = false;

void init_ops(py::module_ &m);

PYBIND11_MODULE(csrc, m) {

    init_ops(m);

    m.def("profiling", [&](bool enable) {
        enable_profile = enable;
        g_queue.update_queue(enable);
    });

    py::class_<tensor>(m, "tensor")
        .def(py::init<>())
        .def(py::init<const py::array&>())
        .def(py::init<const std::vector<cl_uint>&, py::dtype>())
        .def("numpy", &tensor::to_numpy)
        .def_property_readonly("addr", &tensor::addr)
        .def_property("offset", &tensor::get_offset, &tensor::set_offset)
        .def_property_readonly("shape", &tensor::get_shape)
        .def_property_readonly("strides", &tensor::get_strides)
        .def_property_readonly("numel", &tensor::get_numel)
        .def_property_readonly("dtype", &tensor::get_dtype)
        .def(py::pickle(
            // https://docs.python.org/3/library/pickle.html#pickling-class-instances
            [](tensor& p) {  // __getstate__
                return py::make_tuple(p.to_numpy());
            },
            [](py::tuple t) {  // __setstate__
                auto arr = t[0].cast<py::array>();
                if (arr.size() == 0)
                    return tensor();
                return tensor(arr);
            }));

    py::class_<cl_kernels>(m, "kernels")
        .def(py::init<py::bytes, std::string>(), py::arg("bin_bytes"), py::arg("options") = "")
        .def(py::init<std::string, std::string, std::string>(), py::arg("source") = "", py::arg("options") = "", py::arg("dump_dir") = "")
        .def("enqueue", &cl_kernels::enqueue_py)
        .def("SetKernelExecInfo", &cl_kernels::SetKernelExecInfo)
        .def("info", &cl_kernels::info)
        .def(py::pickle(
            [](cl_kernels& p) {  // __getstate__
                return py::make_tuple(p.m_source, p.m_options);
            },
            [](py::tuple t) {  // __setstate__
                return cl_kernels(t[0].cast<std::string>(), t[1].cast<std::string>(), {});
            }));

    // py::class_<cpp_kernels>(m, "cpp_kernels")
    //     .def(py::init<std::string, std::string, std::string>(), py::arg("source") = "", py::arg("options") = "", py::arg("name") = "")
    //     .def("call", &cpp_kernels::call);

    m.def("dev_info", []() {
        py::dict result;
        g_queue.get_device_info(result);
        return result;
    });

    m.def("finish", []() {
        return g_queue.finish();
    });
}
