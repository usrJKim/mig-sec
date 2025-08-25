// prober.cpp
// Compile with: g++ -o prober prober.cpp -lnvidia-ml
// run ./prober output_filename.csv

#include <nvml.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <time.h>    // for clock_gettime, clock_nanosleep, timespec
#include <errno.h>   // for EINTR
#include <signal.h>

volatile sig_atomic_t keep_running = 1;

void handle_signal(int signal){
    keep_running = 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << "<output_filename.csv>\n";
        return 1;
    }

    const char* filename = argv[1];

    // 0) Signal handler
    signal(SIGTERM, handle_signal);
    signal(SIGINT, handle_signal);
    
    // 1) Initialize NVML
    if (nvmlInit() != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML\n";
        return 1;
    }
    nvmlDevice_t device;
    if (nvmlDeviceGetHandleByIndex(0, &device) != NVML_SUCCESS) {
        std::cerr << "Failed to get NVML device handle\n";
        nvmlShutdown();
        return 1;
    }

    // 2) Open output CSV
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Could not open " << filename << " for writing\n";
        nvmlShutdown();
        return 1;
    }
    out << "time_ms,power_w\n";

    std::cout << "Start Probing...\n";

    // 3) Get a baseline timestamp for both CSV and for nanosleep
    auto wall_start = std::chrono::high_resolution_clock::now();

    // Prepare the first absolute wake-up time = now + 1 ms
    struct timespec next;
    clock_gettime(CLOCK_MONOTONIC, &next);

    const long interval_ns = 1'000'000;  // 1 ms in nanoseconds

    // 4) Main polling loop
    while (keep_running) {
        // Query power usage (milliwatts → watts)
        unsigned int power_mW = 0;
        nvmlDeviceGetPowerUsage(device, &power_mW);
        double power_W = power_mW / 1000.0;

        // Compute elapsed wall-clock time in ms
        auto now = std::chrono::high_resolution_clock::now();
        long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - wall_start).count();

        // Write CSV line
        out << ms << "," << power_W << "\n";

        // Flush once a second
        if ((ms % 1000) == 0) out.flush();

        // --- compute next absolute deadline ---
        next.tv_nsec += interval_ns;
        if (next.tv_nsec >= 1'000'000'000) {
            next.tv_sec  += next.tv_nsec / 1'000'000'000;
            next.tv_nsec %= 1'000'000'000;
        }

        // Sleep until that absolute time (retry on EINTR)
        int err;
        do {
            err = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr);
        } while (err == EINTR);
        if (err != 0) {
            std::cerr << "clock_nanosleep failed: " << err << "\n";
            break;
        }
    }

    // Cleanup (unreachable in infinite loop unless error)
    out.close();
    nvmlShutdown();
    return 0;
}
