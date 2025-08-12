// prober4.cpp
// Compile with: g++ -o prober prober4.cpp -lnvidia-ml

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

// Raw collected <time_ms, power_W, temp_C> samples
static std::vector<std::tuple<long long,double,unsigned int>> samples;

// Signal handler: request stop on Ctrl‑C
void handle_sigint(int) {
    stop_requested.store(true);
}

int main(int argc, char* argv[]) {
    std::string output_path = argv[1];
    std::signal(SIGINT, handle_sigint);

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

    // 3) Main sampling loop
    while (!stop_requested.load()) {
        // Drain all new samples
        bufCount = static_cast<unsigned int>(buffer.size());
        ret = nvmlDeviceGetSamples(
            device, sampleType, lastTimestamp,
            &valueType, &bufCount, buffer.data()
        );
        if (ret == NVML_SUCCESS && bufCount > 0) {
            for (unsigned int i = 0; i < bufCount; ++i) {
                lastTimestamp = buffer[i].timeStamp;               // µs
                long long ms   = lastTimestamp / 1000;             // → ms
                double powerW  = buffer[i].sampleValue.uiVal / 1000.0; // mW → W

                // *** New: get current GPU temperature ***
                unsigned int tempC = 0;
                if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &tempC) != NVML_SUCCESS) {
                    std::cerr << "Failed to get temperature\n";
                }

                samples.emplace_back(ms, powerW, tempC);
            }
        }
        // Sleep briefly before polling again
        usleep(10000);  // 10 ms
    }

    // 4) Shut down NVML
    nvmlShutdown();

    if (samples.empty()) {
        std::cerr << "No samples collected.\n";
        return 1;
    }

    // 5) Offset timestamps so the first sample is at t=0
    long long offset = std::get<0>(samples.front());
    for (auto &pr : samples) {
        std::get<0>(pr) -= offset;
    }

    // 6) Stream the filled 1 ms grid to CSV (now including temp_C)
    std::ofstream out(output_path);
    out << "time_ms,power_w,temp_C\n";

    long long t_start = std::get<0>(samples.front());  // == 0 after offset
    long long t_end   = std::get<0>(samples.back());
    size_t    idx     = 0;

    for (long long t = t_start; t <= t_end; ++t) {
        // Advance idx so samples[idx].first is the last sample ≤ t
        while (idx + 1 < samples.size() && std::get<0>(samples[idx+1]) <= t) {
            ++idx;
        }
        double p      = std::get<1>(samples[idx]);
        unsigned int tc = std::get<2>(samples[idx]);
        out << t << ',' << p << ',' << tc << '\n';
    }

    std::cout << "Wrote filled data to power_data.csv ("
              << (t_end - t_start + 1) << " rows)\n";
    return 0;
}
