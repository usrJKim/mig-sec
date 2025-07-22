#include <iostream>
#include <fstream>            // for std::ifstream
#include <string>             // for std::string
#include <vector>             // for std::vector
#include <cstdlib>            // for std::atoi
#include <cmath>              // for __sinf, __expf
#include <chrono>             // for timing
#include <thread>             // for std::thread, sleep_for
#include <cuda_runtime.h>     // for CUDA runtime API

//TODO set TPB to 1 and check results
// Busy‑spin kernel: runs for maxCycles cycles per thread
__global__ void timedSpinKernel(int N, unsigned long long maxCycles) {
    unsigned long long start = clock64();
    double idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    while (clock64() - start < maxCycles) {
        // some non‑trivial work to hold the SM busy
        //idx = idx * __sinf(idx) + __expf(idx);
        idx += idx;
    }
}

// Empty kernel to bump clocks (or hold SM busy very lightly)
__global__ void emptyKernel() { }

// Launch helper: grid of (blocks × TPB) threads, all spinning
void launchPulse(int N, unsigned long long maxCycles, int blocks, int TPB) {
    timedSpinKernel<<<blocks, TPB>>>(N, maxCycles);
    cudaDeviceSynchronize();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <pulse_ms> <delay_ms> <input_file.csv>\n"
                  << "  input_file.csv: comma-separated decimal symbols (0 to totalSM)\n";
        return 1;
    }

    // 1) Parse command-line arguments
    int pulse_ms = std::atoi(argv[1]);
    int delay_ms = std::atoi(argv[2]);
    std::string filename = argv[3];

    // 2) Read symbols from CSV file
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Error: could not open file '" << filename << "'\n";
        return 1;
    }
    std::vector<int> symbols;
    std::string token;
    while (std::getline(in, token, ',')) {
        try {
            symbols.push_back(std::stoi(token));
        } catch (...) {
            std::cerr << "Invalid number in input: '" << token << "'\n";
            return 1;
        }
    }

    // 3) Query GPU properties
    cudaDeviceProp prop;
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA devices found\n";
        return 1;
    }
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties failed: "
                << cudaGetErrorString(err) << "\n";
        return 1;
    }

    int totalSM = prop.multiProcessorCount;
    //int TPB      = prop.maxThreadsPerBlock;
    int TPB = 1;

    // 4) Compute cycles per pulse
    unsigned long long ticks_per_ms = static_cast<unsigned long long>(prop.clockRate);
    unsigned long long maxCycles    = ticks_per_ms * pulse_ms;

    std::cout << "[*] SMs=" << totalSM
              << ", TPB=" << TPB
              << ", pulse=" << pulse_ms
              << " ms => " << maxCycles
              << " cycles\n";

    // 5) Warm up CUDA context (one‑time cost)
    emptyKernel<<<1,1>>>();
    cudaDeviceSynchronize();
    //cudaSetDevice(0);  // Ensure the CUDA context is created

    // 6) Process each symbol
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < symbols.size(); ++i) {
        int v = symbols[i];
        if (v < 0)     v = 0;
        if (v > totalSM) v = totalSM;

        auto pulse_start = std::chrono::high_resolution_clock::now();
        if (v == 0) {
            // idle slot: in background tear down and re-create context + idle kernel
            std::thread reset_thread([](){
                // destroys current CUDA context
                cudaDeviceReset();
                // next CUDA call will rebuild it
                //cudaSetDevice(0);
                //emptyKernel<<<1,1>>>();
                //cudaDeviceSynchronize();
            });
            reset_thread.detach();

            // main thread just sleeps
            std::this_thread::sleep_for(
                std::chrono::milliseconds(delay_ms)
            );
        } else {
            // active slot: launch the busy‑spin pulse
            launchPulse(v * TPB, maxCycles/5, v, TPB);
            launchPulse(v * TPB, maxCycles/5, v, TPB);
            launchPulse(v * TPB, maxCycles/5, v, TPB);
            launchPulse(v * TPB, maxCycles/5, v, TPB);
            launchPulse(v * TPB, maxCycles/5, v, TPB);
            std::thread reset_thread([](){
                // destroys current CUDA context
                cudaDeviceReset();
                // next CUDA call will rebuild it
                //cudaSetDevice(0);
                //emptyKernel<<<1,1>>>();
                //cudaDeviceSynchronize();
            });
            reset_thread.detach();
        }
        auto pulse_end = std::chrono::high_resolution_clock::now();

        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(
            pulse_end - pulse_start
        ).count();

        std::cout << "[*] launchPulse for v=" << v
                    << " took " << dur << " ms\n";
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[*] Total time: " << secs
              << " s => " << (symbols.size() / secs)
              << " symbols/s\n";

    return 0;
}
