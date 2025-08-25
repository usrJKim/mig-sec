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
#include <iomanip>
#include <sstream>

std::string getCurrentTimeString() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto in_time_t = system_clock::to_time_t(now);
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    std::ostringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%H:%M:%S")
       << '.' << std::setw(3) << std::setfill('0') << ms.count();
    return ss.str();
}

// Shared state for clean shutdown
static std::atomic<bool> stop_requested{false};

// Raw collected <time_ms, power_W> samples
static std::vector<std::tuple<long long,double, std::string>> samples;

// Signal handler: request stop on Ctrl‑C
void handle_sigint(int) {
    stop_requested.store(true);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <output_csv_path>\n";
        return 1;
    }
    std::string output_path = argv[1];

    std::signal(SIGINT, handle_sigint);
    std::signal(SIGTERM, handle_sigint);

    // 0) Signal handler

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
    while (!stop_requested.load()) {
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
                std::string now_str = getCurrentTimeString();
                samples.emplace_back(ms, powerW, now_str);
            }
        }
        //else{
        //    std::cout << "No new samples or error: " << nvmlErrorString(ret) << '\n';
        //}
        //if(flag) break;
        // Sleep briefly before polling again
        usleep(20000);  // 20 ms
        //flag = true;
    }

    // 4) Shut down NVML
    nvmlShutdown();

    if (samples.empty()) {
        std::cerr << "No samples collected.\n";
        return 1;
    }

    // 5) Offset timestamps so the first sample is at t=0
    long long offset = std::get<0>(samples.front());
    for (auto &tup : samples) {
        std::get<0>(tup) -= offset;
    }

    // 6) Stream the filled 1 ms grid to CSV
    std::ofstream out(output_path);
    out << "time_ms,power_w,wall_time_str\n";

    long long t_start = std::get<0>(samples.front());  // == 0 after offset
    long long t_end   = std::get<0>(samples.back());
    size_t    idx     = 0;

    for (long long t = t_start; t <= t_end; ++t) {
        // Advance idx so samples[idx].first is the last sample ≤ t
        while (idx + 1 < samples.size() && std::get<0>(samples[idx+1]) <= t) {
            ++idx;
        }
        double time = std::get<0>(samples[idx]);
        double p = std::get<1>(samples[idx]);
        std::string wall = std::get<2>(samples[idx]);
        out << time << ',' << p << ',' << wall <<'\n';
    }

    std::cout << "Wrote filled data to "<<output_path<< "("
              << (t_end - t_start + 1) << " rows)\n";
    return 0;
}
