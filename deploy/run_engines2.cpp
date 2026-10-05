// run_engines2 —— v1 run_engines + CUDA graph (Task 12 / plan B)
// 与 v1 完全兼容的多引擎链 runner；新增 --graph：
//   eager 计时照旧 -> cudaStreamCapture 捕获整链 enqueueV3 -> instantiate ->
//   graph 计时循环，打印第二行 "MEAN ... (graph)"。dump 用最后路径的输出缓冲
//   （graph 与 eager 共享同一组静态地址，输出一致）。
// 捕获失败（个别 tactic/插件含同步）自动回退 eager 并告警，不影响结果正确性。
// 用法: run_engines2 e1.engine [e2.engine ...] [plugin.so]
//            in_dir1 [in_dir2] --dump out_dir [--iters N] [--warmup W] [--graph]
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace nvinfer1;

static class Logger : public ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kERROR) std::cerr << "[TRT] " << msg << "\n";
    }
} g_logger;

static std::map<std::string, std::vector<char>> load_manifest(
        const std::string& dir) {
    std::map<std::string, std::vector<char>> blobs;
    std::ifstream f(dir + "/manifest.tsv");
    if (!f) return blobs;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string name, dtype, dims, file;
        std::getline(ss, name, '\t');
        std::getline(ss, dtype, '\t');
        std::getline(ss, dims, '\t');
        std::getline(ss, file, '\t');
        FILE* b = fopen((dir + "/" + file).c_str(), "rb");
        if (!b) { std::cerr << "missing blob " << dir << "/" << file << "\n";
                  blobs.clear(); return blobs; }
        fseek(b, 0, SEEK_END);
        long sz = ftell(b);
        fseek(b, 0, SEEK_SET);
        std::vector<char> buf(sz);
        if (fread(buf.data(), 1, sz, b) != (size_t)sz) {
            fclose(b); blobs.clear(); return blobs;
        }
        fclose(b);
        blobs[name] = std::move(buf);
    }
    f.close();
    return blobs;
}

static const char* dtype_str(DataType t) {
    switch (t) {
        case DataType::kFLOAT: return "f32";
        case DataType::kHALF: return "f16";
        case DataType::kINT32: return "i32";
        case DataType::kINT8: return "i8";
        case DataType::kBOOL: return "b8";
        default: return "unk";
    }
}

static size_t dtype_size(DataType t) {
    switch (t) {
        case DataType::kFLOAT: return 4;
        case DataType::kHALF: return 2;
        case DataType::kINT32: return 4;
        case DataType::kINT8: return 1;
        case DataType::kBOOL: return 1;
        default: return 0;
    }
}

static long long volume(const Dims& d) {
    long long v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= d.d[i];
    return v;
}

struct TensorBuf {
    void* dev = nullptr;
    size_t bytes = 0;
};

static bool capture_and_time(cudaStream_t stream, int K,
                             std::vector<std::unique_ptr<IExecutionContext>>& ctxs,
                             cudaEvent_t ev0, cudaEvent_t ev1, int iters,
                             cudaStreamCaptureMode mode) {
    cudaStreamBeginCapture(stream, mode);
    for (int k = 0; k < K; ++k) ctxs[k]->enqueueV3(stream);
    cudaGraph_t graph = nullptr;
    cudaError_t cap = cudaStreamEndCapture(stream, &graph);
    if (cap != cudaSuccess || !graph) {
        cudaGetLastError();  // clear capture-invalidation error state
        return false;
    }
    cudaGraphExec_t gexec = nullptr;
    if (cudaGraphInstantiate(&gexec, graph, nullptr, nullptr, 0) != cudaSuccess) {
        cudaGetLastError();
        cudaGraphDestroy(graph);
        return false;
    }
    float total_ms = 0;
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(ev0, stream);
        cudaGraphLaunch(gexec, stream);
        cudaEventRecord(ev1, stream);
        cudaEventSynchronize(ev1);
        float ms;
        cudaEventElapsedTime(&ms, ev0, ev1);
        total_ms += ms;
        if (iters <= 5 || i == 0) std::cout << "graph iter " << i << ": " << ms << " ms\n";
    }
    std::cout << "MEAN " << total_ms / iters << " ms over " << iters
              << " iters (graph)\n";
    cudaGraphExecDestroy(gexec);
    cudaGraphDestroy(graph);
    return true;
}

int main(int argc, char** argv) {
    std::vector<std::string> engPaths;
    const char* pluginSo = nullptr;
    const char* dumpDir = nullptr;
    std::vector<std::string> inDirs;
    int iters = 1, warmup = 0;
    bool useGraph = false, noSync = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--dump" && i + 1 < argc) dumpDir = argv[++i];
        else if (a == "--iters" && i + 1 < argc) iters = atoi(argv[++i]);
        else if (a == "--warmup" && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (a == "--graph") useGraph = true;
        else if (a == "--nosync") noSync = true;
        else if (a.rfind("--", 0) == 0) { std::cerr << "unknown " << a << "\n"; return 2; }
        else if (a.size() > 6 && a.rfind(".engine") == a.size() - 7)
            engPaths.push_back(a);
        else if (!pluginSo && a.find(".so") != std::string::npos)
            pluginSo = argv[i];
        else inDirs.push_back(a);
    }
    if (engPaths.empty() || inDirs.empty()) {
        std::cerr << "用法: " << argv[0] << " e1.engine [e2.engine ...]"
                  " [plugin.so] in_dir1 [in_dir2] --dump out_dir"
                  " [--iters N] [--warmup W] [--graph]\n";
        return 2;
    }
    if (pluginSo && !dlopen(pluginSo, RTLD_NOW)) {
        std::cerr << "dlopen(" << pluginSo << ") failed: " << dlerror() << "\n";
        return 1;
    }
    auto rt = std::unique_ptr<IRuntime>(createInferRuntime(g_logger));

    std::map<std::string, std::vector<char>> inputs;
    for (auto& d : inDirs) {
        auto m = load_manifest(d);
        if (m.empty()) { std::cerr << "bad manifest: " << d << "\n"; return 1; }
        for (auto& kv : m) inputs[kv.first] = std::move(kv.second);
    }
    std::cout << "manifest inputs: " << inputs.size() << "\n";

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int K = (int)engPaths.size();
    std::vector<std::unique_ptr<ICudaEngine>> engs(K);
    std::vector<std::unique_ptr<IExecutionContext>> ctxs(K);
    std::map<std::string, TensorBuf> bufs;      // 全链共享: 名字 -> 缓冲
    std::map<std::string, int> produced_by;     // 输出名 -> 引擎序号
    for (int k = 0; k < K; ++k) {
        std::ifstream f(engPaths[k], std::ios::binary);
        if (!f) { std::cerr << "cannot open " << engPaths[k] << "\n"; return 1; }
        std::string buf((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
        engs[k].reset(rt->deserializeCudaEngine(buf.data(), buf.size()));
        if (!engs[k]) { std::cerr << "deserialize FAILED " << engPaths[k]
                                  << "\n"; return 1; }
        ctxs[k].reset(engs[k]->createExecutionContext());
        if (!ctxs[k]) { std::cerr << "context create FAILED\n"; return 1; }
        std::cout << "== engine " << k << " " << engPaths[k] << " IO "
                  << engs[k]->getNbIOTensors() << "\n";
        for (int i = 0; i < engs[k]->getNbIOTensors(); ++i) {
            const char* nm = engs[k]->getIOTensorName(i);
            auto mode = engs[k]->getTensorIOMode(nm);
            Dims d = engs[k]->getTensorShape(nm);
            DataType dt = engs[k]->getTensorDataType(nm);
            size_t bytes = (size_t)volume(d) * dtype_size(dt);
            auto found = bufs.find(nm);
            if (mode == TensorIOMode::kOUTPUT) {
                if (found != bufs.end()) {
                    std::cerr << "output collision: " << nm << "\n"; return 1;
                }
                TensorBuf tb;
                tb.bytes = bytes;
                if (cudaMalloc(&tb.dev, bytes) != cudaSuccess) {
                    std::cerr << "cudaMalloc out " << nm << " failed\n";
                    return 1;
                }
                bufs[nm] = tb;
                produced_by[nm] = k;
                ctxs[k]->setTensorAddress(nm, tb.dev);
                continue;
            }
            // input
            if (found != bufs.end()) {
                if (found->second.bytes != bytes) {
                    std::cerr << "D2D size mismatch " << nm << "\n"; return 1;
                }
                std::cout << "   in  " << nm << " <- D2D\n";
                ctxs[k]->setTensorAddress(nm, found->second.dev);
                continue;
            }
            auto it = inputs.find(nm);
            if (it == inputs.end()) {
                std::cerr << "input missing blob: " << nm << "\n"; return 1;
            }
            if (it->second.size() != bytes) {
                std::cerr << "input size mismatch " << nm << ": file "
                          << it->second.size() << " want " << bytes << "\n";
                return 1;
            }
            TensorBuf tb;
            tb.bytes = bytes;
            if (cudaMalloc(&tb.dev, bytes) != cudaSuccess) {
                std::cerr << "cudaMalloc in " << nm << " failed\n"; return 1;
            }
            cudaMemcpyAsync(tb.dev, it->second.data(), bytes,
                            cudaMemcpyHostToDevice, stream);
            bufs[nm] = tb;
            ctxs[k]->setTensorAddress(nm, tb.dev);
        }
    }
    cudaStreamSynchronize(stream);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    if (noSync) {
        // 批发射单同步：iter 间不让 host 睡眠（DVFS/发射 gap 定性探针）
        std::vector<cudaEvent_t> ev(iters + 1);
        for (auto& e : ev) cudaEventCreate(&e);
        for (int i = 0; i < warmup; ++i)
            for (int k = 0; k < K; ++k) ctxs[k]->enqueueV3(stream);
        cudaEventRecord(ev[0], stream);
        for (int i = 0; i < iters; ++i) {
            for (int k = 0; k < K; ++k) ctxs[k]->enqueueV3(stream);
            cudaEventRecord(ev[i + 1], stream);
        }
        cudaEventSynchronize(ev[iters]);
        float total_ms = 0;
        for (int i = 0; i < iters; ++i) {
            float ms;
            cudaEventElapsedTime(&ms, ev[i], ev[i + 1]);
            total_ms += ms;
            if (i == 0) std::cout << "nosync iter 0: " << ms << " ms\n";
        }
        std::cout << "MEAN " << total_ms / iters << " ms over " << iters
                  << " iters (nosync)\n";
        for (auto& e : ev) cudaEventDestroy(e);
    } else {
        for (int i = 0; i < warmup; ++i)
            for (int k = 0; k < K; ++k) ctxs[k]->enqueueV3(stream);
        cudaStreamSynchronize(stream);
        float total_ms = 0;
        for (int i = 0; i < iters; ++i) {
            cudaEventRecord(ev0, stream);
            for (int k = 0; k < K; ++k) ctxs[k]->enqueueV3(stream);
            cudaEventRecord(ev1, stream);
            cudaEventSynchronize(ev1);
            float ms;
            cudaEventElapsedTime(&ms, ev0, ev1);
            total_ms += ms;
            if (iters <= 5 || i == 0)
                std::cout << "iter " << i << ": " << ms << " ms\n";
        }
        std::cout << "MEAN " << total_ms / iters << " ms over " << iters
                  << " iters\n";
    }

    if (useGraph) {
        if (!capture_and_time(stream, K, ctxs, ev0, ev1, iters,
                              cudaStreamCaptureModeThreadLocal)) {
            std::cout << "graph capture failed (ThreadLocal), retry Relaxed ...\n";
            if (!capture_and_time(stream, K, ctxs, ev0, ev1, iters,
                                  cudaStreamCaptureModeRelaxed))
                std::cout << "WARN: graph capture unavailable, eager results stand\n";
        }
    }

    if (dumpDir) {
        mkdir(dumpDir, 0755);
        std::ofstream mf(std::string(dumpDir) + "/manifest.tsv");
        for (auto& kv : produced_by) {
            const std::string& nm = kv.first;
            Dims d = engs[kv.second]->getTensorShape(nm.c_str());
            DataType dt = engs[kv.second]->getTensorDataType(nm.c_str());
            size_t bytes = bufs[nm].bytes;
            std::vector<char> host(bytes);
            cudaMemcpy(host.data(), bufs[nm].dev, bytes,
                       cudaMemcpyDeviceToHost);
            std::string fn = nm + ".bin";
            for (auto& ch : fn)
                if (ch == '/') ch = '_';  // 图张量名带 '/'，作文件名会被当路径
            std::ofstream f(std::string(dumpDir) + "/" + fn,
                            std::ios::binary);
            f.write(host.data(), bytes);
            mf << nm << "\t" << dtype_str(dt) << "\t";
            for (int q = 0; q < d.nbDims; ++q)
                mf << (q ? "," : "") << d.d[q];
            mf << "\t" << fn << "\n";
        }
        std::cout << "dumped " << produced_by.size() << " outputs to "
                  << dumpDir << "\n";
    }
    return 0;
}
