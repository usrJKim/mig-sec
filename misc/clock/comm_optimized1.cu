/*
Compile:
  nvcc -o comm_optimized1 comm_optimized1.cu -lcudart
Usage:
  ./comm_optimized1 [workDelayMs] [idleDelayMs] [iterations]
*/

#include <iostream>
#include <thread>
#include <chrono>
#include <spawn.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cstdlib>
#include <cuda_runtime.h>

extern char** environ;
static volatile sig_atomic_t got_sig = 0;

void on_sigusr1(int) {
    got_sig = 1;
}

// A no-op kernel: launching this will ramp SM clocks
__global__ void emptyKernel() {}

// Child logic: launch kernel, wait, signal parent
int child_main(int workDelayMs) {
    // 1) Launch the empty kernel
    emptyKernel<<<1,1>>>();
    //cudaDeviceSynchronize();
    std::cout << "[child] Kernel launched, GPU at high SM clocks\n";

    // 2) Stay busy for workDelayMs
    std::this_thread::sleep_for(std::chrono::milliseconds(workDelayMs));
    std::cout << "[child] Finished work delay: " << workDelayMs << " ms\n";

    // 3) Notify parent
    kill(getppid(), SIGUSR1);
    // 4) Wait to be killed
    pause();
    return 0;
}

int main(int argc, char* argv[]) {
    // Determine mode: child or parent
    if (argc >= 2 && std::string(argv[1]) == "--child") {
        // Child invocation: argv[2] = workDelayMs
        int workDelay = (argc >= 3) ? std::atoi(argv[2]) : 1000;
        return child_main(workDelay);
    }

    // Parent invocation: parse parameters
    int workDelayMs = 1000;
    int idleDelayMs = 1800;
    int iterations   = 10;
    if (argc >= 2) workDelayMs = std::atoi(argv[1]);
    if (argc >= 3) idleDelayMs = std::atoi(argv[2]);
    if (argc >= 4) iterations   = std::atoi(argv[3]);

    // Install SIGUSR1 handler
    struct sigaction sa{};
    sa.sa_handler = on_sigusr1;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGUSR1, &sa, nullptr);

    char* exePath = argv[0];
    for (int i = 0; i < iterations; ++i) {
        got_sig = 0;

        // Prepare child args: [exePath, "--child", workDelayStr, NULL]
        std::string workStr = std::to_string(workDelayMs);
        char* child_argv[] = { exePath, (char*)"--child", const_cast<char*>(workStr.c_str()), nullptr };

        // Spawn child via vfork semantics
        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        pid_t pid;
        int flags = POSIX_SPAWN_USEVFORK;
        int err = posix_spawn(&pid, exePath, &fa, nullptr, child_argv, environ);
        posix_spawn_file_actions_destroy(&fa);
        if (err != 0) {
            std::cerr << "[parent] posix_spawn failed: " << strerror(err) << "\n";
            return 1;
        }
        std::cout << "[parent] Spawned child pid=" << pid << "\n";

        // Wait for child's SIGUSR1
        while (!got_sig) pause();
        std::cout << "[parent] Received signal, killing child pid=" << pid << "\n";
        kill(pid, SIGKILL);
        waitpid(pid, nullptr, 0);

        // Idle delay
        std::this_thread::sleep_for(std::chrono::milliseconds(idleDelayMs));
        std::cout << "[parent] Finished idle delay: " << idleDelayMs << " ms\n";
    }
    return 0;
}
