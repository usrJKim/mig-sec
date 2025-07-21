//nvcc -o empty_reset empty_reset.cu -lnvidia-ml
#include <iostream>
#include <unistd.h>
#include <signal.h>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>

// Plain, non-templated empty kernel
__global__ void emptyKernel() {}

pid_t childPid = -1;
int iterations = 10;            // Number of fork/kill cycles
int childWaitSecs = 1500;         // Child wait before notifying parent

// Signal handler: on SIGUSR1, kill the child immediately
void on_child_ready(int) {
    if (childPid > 0) {
        kill(childPid, SIGKILL);
    }
}

int main(int argc, char** argv) {
    // Optional: parse iterations and wait from args
    if (argc >= 2) iterations    = std::atoi(argv[1]);
    if (argc >= 3) childWaitSecs = std::atoi(argv[2]);

    // Install parent signal handler
    struct sigaction sa{};
    sa.sa_handler = on_child_ready;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGUSR1, &sa, nullptr);

    for (int i = 0; i < iterations; ++i) {
        childPid = fork();
        if (childPid == 0) {
            // ---- Child ----
            // Launch one empty kernel and sync
            std::cout << "empty kernel launched in child process (PID: " << getpid() << ")\n";
            emptyKernel<<<1,1>>>();
            //cudaDeviceSynchronize();
            // Wait before notifying parent
            std::this_thread::sleep_for(std::chrono::milliseconds(childWaitSecs));
            // Notify parent to kill this child
            kill(getppid(), SIGUSR1);
            // Wait to be killed
            pause();
            // (never reached)
        }
        else if (childPid > 0) {
            // ---- Parent ----
            // Wait until child signals readiness
            pause();
            std::cout << "Parent received signal, killing child (PID: " << childPid << ")\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        }
        else {
            std::cerr << "fork() failed" << std::endl;
            break;
        }
    }

    return 0;
}
