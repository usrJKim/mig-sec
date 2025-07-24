// prober3.cpp
// Compile with: g++ -o prober3 prober3.cpp -lnvidia-ml

#include <nvml.h>
#include <chrono>
#include <csignal>
#include <atomic>
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <thread>
#include <unistd.h>  // for usleep

// Shared state for clean shutdown
static std::atomic<bool> stop_requested{false};

// Raw collected <time_ms, power_W> samples
static std::vector<std::pair<long long,double>> samples;

// Signal handler: request stop on Ctrl‑C
void handle_sigint(int) {
    stop_requested.store(true);
}

volatile sig_atomic_t keep_running = 1;

void handle_signal(int signal){
    keep_running = 0;
}

int main() {
    // std::signal(SIGINT, handle_sigint);

    // 0) Signal handler
    signal(SIGTERM, handle_signal);
    signal(SIGINT, handle_signal);

    // 1) Initialize NVML and get GPU handle
    if (nvmlInit() != NVML_SUCCESS) {
        std::cerr << "nvmlInit() failed\n";
        return 1;
    }
    nvmlDevice_t device;
    if (nvmlDeviceGetHandleByIndex(0, &device) != NVML_SUCCESS) {
        std::cerr << "nvmlDeviceGetHandleByIndex(0) failed\n";
        nvmlShutdown();
        return 1;
    }

    std::cout << "Sampling power via nvmlDeviceGetSamples()... Ctrl‑C to stop\n";

    // 2) Prepare to sample
    nvmlSamplingType_t  sampleType    = NVML_TOTAL_POWER_SAMPLES;
    nvmlValueType_t     valueType;
    unsigned int        bufCount      = 0;
    unsigned long long  lastTimestamp = 0;

    // First, query how many entries NVML has buffered
    nvmlReturn_t ret = nvmlDeviceGetSamples(
        device, sampleType, lastTimestamp,
        &valueType, &bufCount, nullptr
    );
    if (ret != NVML_SUCCESS || bufCount == 0) {
        std::cerr << "Failed to query sample buffer or no samples available\n";
        nvmlShutdown();
        return 1;
    }
    std::cout << "Buffer size: " << bufCount << " samples\n";

    // Allocate a buffer for nvmlSample_t
    std::vector<nvmlSample_t> buffer(bufCount);
    //bool flag = false;
    // 3) Main sampling loop
    while (keep_running) {
    // while (!stop_requested.load()) {
        // Drain all new samples
        bufCount = static_cast<unsigned int>(buffer.size());
        ret = nvmlDeviceGetSamples(
            device, sampleType, lastTimestamp,
            &valueType, &bufCount, buffer.data()
        );
        if (ret == NVML_SUCCESS && bufCount > 0) {
            //std::cout << "Collected " << bufCount << " samples\n";
            for (unsigned int i = 0; i < bufCount; ++i) {
                lastTimestamp = buffer[i].timeStamp;               // µs
                long long ms   = lastTimestamp / 1000;             // → ms
                double powerW  = buffer[i].sampleValue.uiVal / 1000.0; // mW → W
                samples.emplace_back(ms, powerW);
            }
        }
        //else{
        //    std::cout << "No new samples or error: " << nvmlErrorString(ret) << '\n';
        //}
        //if(flag) break;
        // Sleep briefly before polling again
        usleep(10000);  // 10 ms
        //flag = true;
    }

    // 4) Shut down NVML
    nvmlShutdown();

    if (samples.empty()) {
        std::cerr << "No samples collected.\n";
        return 1;
    }

    // 5) Offset timestamps so the first sample is at t=0
    long long offset = samples.front().first;
    for (auto &pr : samples) {
        pr.first -= offset;
    }

    // 6) Stream the filled 1 ms grid to CSV
    std::ofstream out("power_data.csv");
    out << "time_ms,power_w\n";

    long long t_start = samples.front().first;  // == 0 after offset
    long long t_end   = samples.back().first;
    size_t    idx     = 0;

    for (long long t = t_start; t <= t_end; ++t) {
        // Advance idx so samples[idx].first is the last sample ≤ t
        while (idx + 1 < samples.size() && samples[idx+1].first <= t) {
            ++idx;
        }
        double p = samples[idx].second;
        out << t << ',' << p << '\n';
    }

    std::cout << "Wrote filled data to power_data.csv ("
              << (t_end - t_start + 1) << " rows)\n";
    return 0;
}
