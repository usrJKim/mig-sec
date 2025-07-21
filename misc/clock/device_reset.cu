// nvcc -o device_reset device_reset.cu

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>

// A no-op kernel: launching this will still trigger the GPU to ramp to max SM clocks
__global__ void emptyKernel() {}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <binary sequence, e.g. 0,1,0,1>" \
                  << " [delay_ms, e.g. 500]\n";
        return 1;
    }

    // Optional delay between operations (ms)
    int delayMs = 100;
    if (argc >= 3) {
        delayMs = std::atoi(argv[2]);
    }

    // Parse the comma-separated binary sequence
    std::string input = argv[1];
    std::istringstream ss(input);
    std::string token;
    std::vector<int> bits;
    while (std::getline(ss, token, ',')) {
        if (token == "0") bits.push_back(0);
        else if (token == "1") bits.push_back(1);
        else {
            std::cerr << "Invalid bit '" << token << "'. Use only '0' or '1'.\n";
            return 1;
        }
    }

    // Process each bit: 0 = reset device, 1 = launch empty kernel
    for (size_t i = 0; i < bits.size(); ++i) {
        if (bits[i] == 0) {
            cudaDeviceReset();
            std::cout << "[+] bit " << i << " = 0: called cudaDeviceReset()\n";
        } else {
            emptyKernel<<<1, 1>>>();
            std::cout << "[+] bit " << i << " = 1: launched empty kernel\n";
        }
        // Delay between operations
        std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
        std::cout << "[+] Delay of " << delayMs << " ms before next bit\n";
    }

    std::cout << "[+] Sequence processing complete. Exiting.\n";
    return 0;
}
