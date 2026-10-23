
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.h>

#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <iostream>

#ifndef ASSERT

template <typename... TS>
void _write_all(std::ostream& os, TS&&... args) {
    int dummy[sizeof...(TS)] = {(os << std::forward<TS>(args), 0)...};
    (void)dummy;
}

#ifdef __x86_64__
#    define TRAP_INST() __asm__("int3");
#endif

#ifdef __aarch64__
#    define TRAP_INST() __asm__("brk #0x1");
#endif

#define ASSERT(cond, ...)                                                                   \
        if (!(cond)) {                                                                      \
            std::stringstream ss;                                                           \
            _write_all(ss, __FILE__, ":", __LINE__, " ", #cond, " failed:", ##__VA_ARGS__); \
            std::cout << "\033[31m" << ss.str() << "\033[0m" << std::endl;                  \
            /* TRAP_INST();*/                                                               \
            throw std::runtime_error(ss.str());                                             \
        }

#endif

template <typename F>
struct ext_func_t {
    #if defined(__cplusplus) && __cplusplus < 201703L
    inline int uncaught_exceptions() {
        return (int)std::uncaught_exception();
    }

    template <class F, class... ArgTypes>
    using invoke_result = std::result_of<F(ArgTypes...)>;
    #else

    inline int uncaught_exceptions() {
        return std::uncaught_exceptions();
    }

    template <class FUNC, class... ArgTypes>
    using invoke_result = std::invoke_result<FUNC, ArgTypes...>;
    #endif

    ext_func_t(const char *ext_func_name, const char *vendor_name = "Intel")
        : ext_func_ptrs_(vendor_platforms(vendor_name).size()) {
        for (size_t i = 0; i < vendor_platforms(vendor_name).size(); ++i) {
            auto p = vendor_platforms(vendor_name)[i];
            auto it = ext_func_ptrs_.insert(
                    {p, load_ext_func(p, ext_func_name)});
            assert(it.second);
        }
    }

    template <typename... Args>
    typename invoke_result<F, Args...>::type operator()(
            cl_platform_id platform, Args... args) const {
        auto f = get_func(platform);
        return f(args...);
    }

    // F get_func(engine_t *engine) const {
    //     return get_func(get_platform(engine));
    // }

    F get_func(cl_platform_id platform) const {
        return ext_func_ptrs_.at(platform);
    }

private:
    std::unordered_map<cl_platform_id, F> ext_func_ptrs_;

    static F load_ext_func(cl_platform_id platform, const char *ext_func_name) {
        return reinterpret_cast<F>(clGetExtensionFunctionAddressForPlatform(
                platform, ext_func_name));
    }

    static const std::vector<cl_platform_id> &vendor_platforms(
            const char *vendor_name) {
        static auto vendor_platforms = get_vendor_platforms(vendor_name);
        return vendor_platforms;
    }

    static std::vector<cl_platform_id> get_vendor_platforms(
            const char *vendor_name) {
        cl_uint num_platforms = 0;
        cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
        if (err != CL_SUCCESS) return {};

        std::vector<cl_platform_id> platforms(num_platforms);
        err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        if (err != CL_SUCCESS) return {};

        std::vector<cl_platform_id> vendor_platforms;
        char platform_vendor_name[128] = {};
        for (cl_platform_id p : platforms) {
            err = clGetPlatformInfo(p, CL_PLATFORM_VENDOR,
                    sizeof(platform_vendor_name), platform_vendor_name,
                    nullptr);
            if (err != CL_SUCCESS) continue;
            if (std::string(platform_vendor_name).find(vendor_name)
                    != std::string::npos)
                vendor_platforms.push_back(p);
        }

        // OpenCL can return a list of platforms that contains duplicates.
        std::sort(vendor_platforms.begin(), vendor_platforms.end());
        vendor_platforms.erase(
                std::unique(vendor_platforms.begin(), vendor_platforms.end()),
                vendor_platforms.end());
        return vendor_platforms;
    }
};




template<typename T>
T getDeviceInfo(cl_device_id device, cl_device_info param_name) {
    size_t bytes = 0;
    clGetDeviceInfo(device, param_name, 0, nullptr, &bytes);
    ASSERT(bytes == sizeof(T), "clGetDeviceInfo needs more space");
    T ret;
    clGetDeviceInfo(device, param_name, sizeof(ret), &ret, nullptr);
    return ret;
}

template<>
inline std::string getDeviceInfo<std::string>(cl_device_id device, cl_device_info param_name) {
    size_t bytes = 0;
    clGetDeviceInfo(device, param_name, 0, nullptr, &bytes);
    std::string ret(bytes, 0);
    clGetDeviceInfo(device, param_name, bytes, &ret[0], nullptr);
    return ret;
}

struct ocl_queue {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    bool enable_profile;
    std::vector<cl_event> events;

    ocl_queue() {
        select_device();
        update_queue(true);
    }

    std::vector<uint64_t> finish() {
        ASSERT(clFinish(queue) == CL_SUCCESS);
        std::vector<uint64_t> ret;
        if (enable_profile) {
            for (auto& evt : events) {
                ret.emplace_back(0);
                if (evt != 0) {
                    cl_ulong start, end;
                    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
                    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
                    ret.back() = end - start;
                }
            }
        }
        events.clear();
        return ret;
    }

    void* malloc_host(size_t sz) {
        using clHostMemAllocINTEL_func_t = void *(*)(cl_context, const cl_ulong *,
                size_t, cl_uint, cl_int *);
        static ext_func_t<clHostMemAllocINTEL_func_t> ext_func(
                "clHostMemAllocINTEL");
        cl_int err;
        return ext_func(platform, context, nullptr, sz, 0, &err);
        //return clSVMAlloc(context, CL_MEM_READ_WRITE, sz, 64);
    }
    void* malloc_device(size_t sz) {
        using clDeviceMemAllocINTEL_func_t = void *(*)(cl_context, cl_device_id,
            cl_ulong *, size_t, cl_uint, cl_int *);
        static ext_func_t<clDeviceMemAllocINTEL_func_t> ext_func(
                "clDeviceMemAllocINTEL");
        cl_int err;
        return ext_func(platform, context, device, nullptr, sz, 0, &err);
        //return clSVMAlloc(context, CL_MEM_READ_WRITE, sz, 64);
    }
    void memcpy_HtoD(void* pdst, void* psrc, size_t bytes) {
        using clEnqueueMemcpyINTEL_func_t
                = cl_int (*)(cl_command_queue, cl_bool, void *, const void *,
                        size_t, cl_uint, const cl_event *, cl_event *);
        static ext_func_t<clEnqueueMemcpyINTEL_func_t> ext_func(
                "clEnqueueMemcpyINTEL");

        ASSERT(clFinish(queue) == CL_SUCCESS);
        ASSERT(ext_func(platform, queue, CL_TRUE, pdst, psrc, bytes, 0, nullptr, nullptr) == CL_SUCCESS);
        // clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, pdst, bytes, 0, nullptr, nullptr);
        // std::memcpy(pdst, psrc, bytes);
        // clEnqueueSVMUnmap(queue, pdst, 0, nullptr, nullptr);
        ASSERT(clFinish(queue) == CL_SUCCESS);
    }

    void memcpy_DtoH(void* pdst, void* psrc, size_t bytes) {
        memcpy_HtoD(pdst, psrc, bytes);
        // ASSERT(clFinish(queue) == CL_SUCCESS);
        // clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, psrc, bytes, 0, nullptr, nullptr);
        // std::memcpy(pdst, psrc, bytes);
        // clEnqueueSVMUnmap(queue, psrc, 0, nullptr, nullptr);
        // ASSERT(clFinish(queue) == CL_SUCCESS);
    }

    template<class MAP>
    void get_device_info(MAP& result) {
        result["CL_DEVICE_NAME"] = getDeviceInfo<std::string>(device, CL_DEVICE_NAME);
        result["CL_DEVICE_EXTENSIONS"] = getDeviceInfo<std::string>(device, CL_DEVICE_EXTENSIONS);
        result["CL_DEVICE_MAX_COMPUTE_UNITS"] = getDeviceInfo<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS);
        result["CL_DEVICE_MAX_CLOCK_FREQUENCY"] = getDeviceInfo<cl_uint>(device, CL_DEVICE_MAX_CLOCK_FREQUENCY);
        result["CL_DEVICE_LOCAL_MEM_SIZE"] = getDeviceInfo<cl_ulong>(device, CL_DEVICE_LOCAL_MEM_SIZE);
        result["CL_DEVICE_MAX_WORK_GROUP_SIZE"] = getDeviceInfo<cl_ulong>(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);

    }

    void select_device(std::vector<std::string> exts = {}) {
        cl_int error;
        cl_uint n = 0;
        clGetPlatformIDs(0, nullptr, &n);
        std::vector<cl_platform_id> ids(n);
        clGetPlatformIDs(n, ids.data(), nullptr);
        int gpu_dev_index = 0;
        platform = 0;
        std::vector<std::pair<cl_device_id, cl_platform_id>> device_ids;
        for(int i = 0; i < n; i++) {
            auto platform_id = ids[i];
            cl_uint n_devs = 0;
            clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, nullptr, &n_devs);
            if (n_devs == 0) continue;

            std::vector<cl_device_id> dev_ids(n_devs);
            error = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, n_devs, dev_ids.data(), nullptr); ASSERT(error == CL_SUCCESS);

            for(int k = 0; k < n_devs; k++) {
                auto devid = dev_ids[k];
                auto dev_ext = getDeviceInfo<std::string>(devid, CL_DEVICE_EXTENSIONS);
                std::cout << " GPU device [" << gpu_dev_index << "] : "
                          << getDeviceInfo<std::string>(devid, CL_DEVICE_NAME)
                          << " by \"" << getDeviceInfo<std::string>(devid, CL_DEVICE_VENDOR) << "\" : "
                          << getDeviceInfo<cl_uint>(devid, CL_DEVICE_MAX_COMPUTE_UNITS) << " EUs "
                          << getDeviceInfo<cl_uint>(devid, CL_DEVICE_MAX_CLOCK_FREQUENCY) << " MHz "
                          << getDeviceInfo<cl_ulong>(devid, CL_DEVICE_LOCAL_MEM_SIZE) << " bytes SLM "
                          << (dev_ext.find("cl_intel_subgroup_matrix_multiply_accumulate") != std::string::npos ? "[DPAS]":"")
                          << (dev_ext.find("cl_intel_subgroup_split_matrix_multiply_accumulate") != std::string::npos ? "[DPASW]":"")
                          << (dev_ext.find("cl_intel_split_work_group_barrier") != std::string::npos ? "[sbarrier]":"")
                          << std::endl;
                device_ids.push_back({devid, platform_id});
                gpu_dev_index++;
            }
        }
        if (device_ids.empty()) {
            throw std::runtime_error("clops: no OpenCL GPU device found");
        }
        device = device_ids[0].first;
        platform = device_ids[0].second;
        context = clCreateContext(NULL, 1, &device, NULL, NULL, &error); ASSERT(error == 0);
        return;
    }

    void update_queue(bool _enable_profile = false) {
        enable_profile = _enable_profile;
        cl_queue_properties queue_properties[] = {
                    CL_QUEUE_PROPERTIES,
                    enable_profile ? CL_QUEUE_PROFILING_ENABLE : 0ul,
                    0ul};

        cl_int error;
        queue = clCreateCommandQueueWithProperties(context, device, queue_properties, &error); ASSERT(error == 0);
    }

    template<typename RET, typename... Args>
    RET call_ext(const char * func_name, Args&&... args) {
        using ext_func_t = RET (*)(Args...);
        ext_func_t ext_func = reinterpret_cast<ext_func_t>(clGetExtensionFunctionAddressForPlatform(platform, func_name));
        ASSERT(ext_func != nullptr, "clGetExtensionFunctionAddressForPlatform()", func_name, " returns nullptr");
        return (*ext_func)(std::forward<Args>(args)...);
    }
};


// composite of cl::Buffer & layout information
// like numpy array
struct tensor {
    std::vector<cl_uint> shape;
    std::vector<cl_uint> strides;
    cl_uint numel = 0;
    std::shared_ptr<void> p_buff;
    py::dtype dt;
    int offset = 0;

    tensor() {}

    ~tensor() = default;

    const std::vector<cl_uint>& get_shape() const {
        return shape;
    }
    const std::vector<cl_uint>& get_strides() const {
        return strides;
    }
    cl_uint get_numel() const {
        return numel;
    }
    py::dtype get_dtype() const {
        return dt;
    }

    template <class T>
    operator T*() const {
        //if (py::dtype::of<T>() != dt)
        //    throw std::runtime_error(std::string("unable to cast from tensor of dtype ") + dt.kind() + " to " + py::dtype::of<T>().kind());
        return reinterpret_cast<T*>(data());
    }

    operator void*() const {
        return data();
    }

    void* data() const {
        return reinterpret_cast<int8_t*>(p_buff.get()) + offset;
    }

    uintptr_t addr() const {
        return reinterpret_cast<uintptr_t>(p_buff.get()) + offset;
    }

    void set_offset(int off) {
        offset = off;
    }
    int get_offset() {
        return offset;
    }

    void resize(const std::vector<cl_uint>& dims, py::dtype dtype);

    tensor(const py::array& arr) {
        resize(arr);
    }
    tensor(const std::vector<cl_uint>& dims, py::dtype dtype) {
        resize(dims, dtype);
    }
    void update_buff();
    void resize(const py::array& b);
    py::array to_numpy();
};

struct cl_kernels {
    std::map<std::string, cl_kernel> kernel_map;
    cl_program m_prog;
    std::string m_source;
    std::string m_options;
    cl_kernels() = default;

    void check_error(const char* name, cl_int error, bool show_build_log = false);
    cl_kernels(py::bytes bin_bytes, std::string options);
    cl_kernels(std::string source, std::string options, std::string dump_dir);

    void setup(std::string source, std::string options, std::string dump_dir);

    cl_kernel get_kernel(std::string kernel_name);
    void set_arg(cl_kernel kernel, int idx, int arg);
    void set_arg(cl_kernel kernel, int idx, float arg);
    void set_arg(cl_kernel kernel, int idx, void* arg);

    int SetKernelExecInfo(std::string kernel_name, int i, bool enable);

    template<typename... Ts>
    void set_args(cl_kernel kernel, Ts... args) {
        int arg_id = 0;
        // since initializer lists guarantee sequencing, this can be used to
        // call a function on each element of a pack, in order:
        int dummy[sizeof...(Ts)] = {(set_arg(kernel, arg_id++, args), 0)...};
    }

    void enqueue(cl_kernel kernel, const std::vector<size_t>& global_size, const std::vector<size_t>& local_size);

    void enqueue_py(std::string kernel_name, const std::vector<size_t>& global_size, const std::vector<size_t>& local_size, py::args args);
    std::string info(const char* kernel_name, const std::vector<int>& local_size, size_t sub_groups);
};


// all jit-based/performance-aware function should be a functor/callable because:
//   - it needs to hold reference to kernel (to save build time & resources)
//   - it needs to do other compile time preparation work and hold the relevant
//     runtime-data-struct (to make runtime faster)
// to optimze compile-time-workload itself, the functor instance itself should be
// cached with compile-time parameter as the key.
//
// because it's a functor, which supposed to have no states, so cache-factory should
// always return shared_ptr to constant object, so it won't behave differently when being
// called by different caller, and this also ensure it's multi-threading safe since it
// won't modify it's content.
//
template <typename... TTypes>
class tuple_hasher {
private:
    typedef std::tuple<TTypes...> Tuple;
    template <int N>
    size_t hash(Tuple& value) const {
        return 0;
    }
    template <int N, typename THead, typename... TTail>
    size_t hash(Tuple& value) const {
        constexpr int Index = N - sizeof...(TTail) - 1;
        return std::hash<THead>()(std::get<Index>(value)) ^ hash<N, TTail...>(value);
    }

public:
    size_t operator()(Tuple value) const {
        auto hv = hash<sizeof...(TTypes), TTypes...>(value);
        return hv;
    }
};

// create const object with internal cache with constructor-args as the key
// this helps reduces construction time overhead, and perfectly suitable
// for caching functor/callable.
template <class T, typename... CArgs>
std::shared_ptr<const T> make_cacheable(CArgs... cargs) {
    std::shared_ptr<const T> sptr;
    auto key = std::make_tuple(cargs...);
    static std::unordered_map<decltype(key), std::weak_ptr<const T>, tuple_hasher<CArgs...>> cache;
    static std::mutex mutex;
    std::lock_guard<std::mutex> guard(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        auto& wptr = it->second;
        sptr = wptr.lock();
        if (!sptr) {
            sptr = std::make_shared<T>(cargs...);
            // ECOUT("make_cacheable re-constructed: ", typeid(T).name(), "(", cargs..., ")");
            wptr = sptr;
        }
    } else {
        sptr = std::make_shared<T>(cargs...);
        // ECOUT("make_cacheable constructed: ", typeid(T).name(), "(", cargs..., ")");
        cache.emplace(std::make_pair(key, std::weak_ptr<const T>(sptr)));
    }
    return sptr;
}