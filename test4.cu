#include <iostream>
#include <fstream>            // for std::ifstream
#include <string>             // for std::string
#include <vector>             // for std::vector
#include <cstdlib>            // for std::atoi
#include <cmath>              // for __sinf
#include <thread>             // only for fighter includes
#include <chrono>             // for std::chrono::high_resolution_clock
#include <time.h>             // for clock_nanosleep, timespec, CLOCK_MONOTONIC
#include <errno.h>            // for EINTR
#include <cuda_runtime.h>     // for CUDA runtime API

// Busy-spin kernel: runs for maxCycles cycles per thread
// Now operates on float data, multiplies by __sinf(val) and adds idx each loop.
__global__ void timedSpinKernel(float* data, int N, unsigned long long maxCycles) {
    unsigned long long start = clock64();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float val = data[idx];
    while (clock64() - start < maxCycles) {
        val *= __sinf(val);
        val += idx;
    }
    data[idx] = val;
}

__global__ void emptyKernel() { }

// Launch helper: grid of (blocks × TPB) threads, all spinning
void launchPulse(float* d_data, int N, unsigned long long maxCycles, int blocks, int TPB) {
    timedSpinKernel<<<blocks, TPB>>>(d_data, N, maxCycles);
    cudaDeviceSynchronize();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <pulse_ms> <delay_ms> <input_file.csv>\n"
                  << "  input_file.csv: comma-separated decimal symbols (0 to totalSM)\n";
        return 1;
    }

    int pulse_ms = std::atoi(argv[1]);
    int delay_ms = std::atoi(argv[2]);
    std::string filename = argv[3];

    // 1) Read symbols from CSV file
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

    // 2) Query GPU SM count and clock rate
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int totalSM = prop.multiProcessorCount;
    const int TPB = prop.maxThreadsPerBlock;

    // 3) Prepare a device buffer for the worst case (full occupancy)
    int maxThreads = totalSM * TPB;
    std::vector<float> h_data(maxThreads);
    for (int i = 0; i < maxThreads; ++i) {
        h_data[i] = static_cast<float>(i);
    }
    float* d_data = nullptr;
    cudaMalloc(&d_data, maxThreads * sizeof(float));
    cudaMemcpy(d_data, h_data.data(), maxThreads * sizeof(float), cudaMemcpyHostToDevice);

    unsigned long long ticks_per_ms = static_cast<unsigned long long>(prop.clockRate);
    unsigned long long maxCycles    = ticks_per_ms * pulse_ms;

    std::cout << "[*] SMs=" << totalSM
              << ", TPB="   << TPB
              << ", pulse=" << pulse_ms
              << " ms => "  << maxCycles
              << " cycles\n";

    // 4) Process each symbol: 0=>idle, v=>launch on v SMs
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < symbols.size(); ++i) {
        int v = symbols[i];
        if (v < 0) v = 0;
        if (v > totalSM) v = totalSM;

        if (v == 0) {
            struct timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            
            // instead of pure sleep, launch a 1‑thread no‑op
            emptyKernel<<<1,1>>>();
            cudaDeviceSynchronize();

            // then wait until the next 100 ms deadline
            ts.tv_sec  += delay_ms / 1000;
            ts.tv_nsec += (delay_ms % 1000) * 1000000;
            if (ts.tv_nsec >= 1000000000) {
                ts.tv_sec++;
                ts.tv_nsec -= 1000000000;
            }
            while (true) {
                int err = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr);
                if (err == 0) break;
                if (err != EINTR) {
                    std::cerr << "clock_nanosleep failed: " << err << "\n";
                    break;
                }
            }
        } else {
            launchPulse(d_data, v * TPB, maxCycles, v, TPB);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[*] Total time: " << secs
              << " s => " << (symbols.size() / secs)
              << " symbols/s\n";

    // Cleanup
    cudaFree(d_data);
    return 0;
}
