// https://github.com/boris-dimitrov/z4_planar_langford_multigpu
//
// Copyright 2017 Boris Dimitrov, Portola Valley, CA 94028.
// Questions? Contact http://www.facebook.com/boris
//
// This program counts all permutations of the sequence 1, 1, 2, 2, 3, 3, ..., n, n
// in which the two occurrences of each m are separated by precisely m other numbers,
// and lines connecting all (m, m) pairs can be drawn on the page without crossing.
//
// See http://www.dialectrix.com/langford.html ("Planar Solutions") or Knuth volume 4a
// page 3.  Todo: Provide better Knuth reference.
//
//
// COMPILE ON LINUX WITH CUDA 8.0:
//
//     nvcc -O5 -o planar_langford_multigpu -std=c++11 -DNDEBUG planar_langford_multigpu.cu
//
// The resulting executable will be named "planar_langford_multigpu".
//
//
// THE ALGORITHM
//
// This program runs a depth-first-search aka backtracking algorithm which chooses
// to "open" or "close" a pair at each position, starting with position 0, and
// whether that pair would be connected from "below" or "above".  There are 4 choices
// for each of the 2*n positions, making it O(4^(2n)) with maximum stack depth 6*n.
//
// When choosing to "close" at position k, it locates the matching "open" at k',
// and computes the distance m = k - k' + 1.  If this m has already been placed,
// closing at position k is not possible.
//
// When the number of open pairs reaches n, opening new pairs is no longer possible.
// Observing this constraint greatly prunes the search tree.
//
// The matching "open" at k' is very easy to find using two auxiliary stacks of
// currently open pairs, one for "below" and one for "above".
//
//
// DEDUPLICATION
//
// To dedup the Left <-> Right reversal symmetry, (1, 1) is placed in pos <= n.
//
// Many, but not all, top <-> bottom twins are deduped by forcing the pair in
// position 0 to be connected from below.
//
// Remaining duplicates are eliminated by storing all solutions in memory,
// with a final sort and count.  Fortunately, the number of solutions to the
// planar Langford problem is quite small, so this is feasible.
//
//
// IMPLEMENTATION TRICKS
//
// A nice boost in performance is realized through the use of a single 64-bit
// integer to encode the positions of *all* currently open pairs;  in this compact
// representation, we can quickly "pop" the position of the most recently open pair
// by using the operations
//
//     ffsl(x)       if x is non-zero, return one plus the index of the least
//                   significant bit of x;  if x is zero, return zero;
//                   result range 0..64
//
//     x &= (x-1)    clear the least signifficant 1-bit of x
//
//
// EXAMPLE OUTPUT
//
//     1488034642458 Computing PL(2, 19)
//     1488034642458 Will use 2 system GPU(s).
//     1488034642463 GPU 1 init took 0.005 seconds.
//     1488034642463 GPU 0 init took 0.005 seconds.
//     1488034652163 GPU 0 computation took 9.70008 seconds on GPU clock, 9.7 seconds on host clock.
//     1488034652178 GPU 1 computation took 9.71449 seconds on GPU clock, 9.715 seconds on host clock.
//     1488034652229 Result 2384 for n = 19 MATCHES previously published result
//
// The first number on each output line is a unix timestamp, i.e., milliseconds elapsed
// since Jan 1, 1970 GMT.  You may convert it to human-readable datetime using python,
// as follows.
//
// 1) Start "python"
// 2) Type "import time" and press enter
// 3) Type "time.localtime(1488034652163 / 1000.0)" and press enter
//
// The result is a decoding of unix timestamp 1488034652163 in your local time zone:
//
//     time.struct_time(tm_year=2017, tm_mon=2, tm_mday=25, tm_hour=6,
//         tm_min=57, tm_sec=32, tm_wday=5, tm_yday=56, tm_isdst=0)
//
//
// PRINTING ALL SOLUTION SEQUENCES
//
// If you wish all solutions sequences printed, change 'kPrint' below to 'true',
// and recompile.
//
//
// ACHIEVEMENTS
//
// On March 6, 2017 at 11:38am PST this program computed PL(2, 28) after very close to 168 hours
// of work on a pair of NVIDIA Titan X Pascal GPUs in a single workstation.
//
//     1488221331811 Computing PL(2, 28)
//     1488221331811 Will use 2 system GPU(s).
//     1488221331817 GPU 1 init took 0.006 seconds.
//     1488221331817 GPU 0 init took 0.006 seconds.
//     1488823122601 GPU 1 computation took 601769 seconds on GPU clock, 601791 seconds on host clock.
//     1488829105991 GPU 0 computation took 607752 seconds on GPU clock, 607774 seconds on host clock.
//     1488829106231 Result 817717 for n = 28 MATCHES previously published result
//
// The GPU clock ran at ~1835 MHz, with each GPU consuming ~150 watts, at temperature ~62 Celsius,
// during most of that 7 day computation.  The last few hours the chips were less busy while
// finishing up a few of the longest running threads, and consumed correspondingly less power.
//
// SEE ALSO
//
// There is a variant of this program that runs on CPUs.  Performance is comparable between a single
// Titan X Pascal GPU (16nm process, mid 2016) and a 22-core Xeon E5-2699v4 (14nm, early 2016).
//
//     https://github.com/boris-dimitrov/z4_planar_langford

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <mutex>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <stdlib.h>
using namespace std;

// to avoid integer overflow, n should not exceed this constant
constexpr int kMaxN = 31;

// kLimit = 100 million means we can use up to 3.2GB of RAM on each GPU to store
// results before sorting;  if more memory is needed, we detect and bail
constexpr int64_t kLimit = 100000000;

// set to "true" if you want each solution printed
constexpr bool kPrint = false;

static_assert(sizeof(int64_t) == 8, "int64_t is not 8 bytes");
static_assert(sizeof(int32_t) == 4, "int32_t is not 4 bytes");
static_assert(sizeof(int8_t) == 1, "int64_t is not 1 byte");

constexpr int64_t lsb = 1;
constexpr int32_t lsb32 = 1;

constexpr int div_up(int p, int q) {
    return (p + (q - 1)) / q;
};

// This type represents a solution by letting pos[m-1] be
// the position of the closing m, for m = 1, 2, ... n.
template <int n>
using Positions = array<int8_t, n>;

// All planar sequences (including duplicates) are stored in a vector for sorting.
template <int n>
using Results = vector<Positions<n>>;

// 8-byte alignment probably helps with concurrent writes to adjacent instances
template <int n>
using PositionsGPUAligned = int64_t[(n + 7) / 8];

template <int n>
using PositionsGPU = int8_t[div_up(n, 8) * 8];

// at depth k in the search, the bits of availability[k+1] represent the still
// unused m;  thus helping ensure that each (m, m) pair is placed just once
template <int n>
using Availability = int32_t[2 * n + 1];

// at depth k in the search, the bits of open[2*k+2] represent the positions
// of all pairs that are open above;  open[2*k+3] is the same for below;
// k ranges over 0, 1, ..., 2n-1
template <int n>
using Open = int64_t[4 * n + 2];

template <int n>
using Stack = int8_t[24 * n];

template <int n>
void print(const Positions<n>& pos);

// For some reason, 4 is the magic number that gives us best perf in this algorithm
constexpr int kThreadsPerBlock = 4;

// subdivide the search tree for this many logical threads
// due to the naive math below, numbers like 2^r-1 work much better than 2^r
constexpr int kNumLogicalThreads = 16383;

// use at most this many of the system's GPUs
constexpr int kMaxDevices = 8;

// To do:  Run on CPU (right now only runs on GPU).
template <int n>
__device__
void dfs(int64_t* p_result,
         Availability<n> &availability,
         Open<n> &open,
         Stack<n> &stack,
         PositionsGPUAligned<n>& pgpualigned,
         const int32_t logical_thread_index,
         mutex* p_mtx)
{
    constexpr int two_n = 2 * n;
    constexpr int64_t msb = lsb << (int64_t)(n - 1);
    constexpr int64_t nn1 = lsb << (2 * n - 1);
    PositionsGPU<n> &pos = *((PositionsGPU<n>*)(&pgpualigned[0]));
    // initially none of the numbers 1, 2, ..., n have been placed;
    // this is represented by setting bits 0..n-1 to 1 in avail
    availability[0] = msb | (msb - 1);
    open[0] = 0;
    open[1] = 0;
    int top = 0;
    int8_t k, m, d, num_open;
    // The following "push" and "pop" should be lambdas, but unfortunately Cuda C++ does not
    // yet support reference capture in lambdas that can run on both CPU and GPU.
    // Hoping for a compiler fix soon.
    #define push(k, m, d, num_open) do { \
        stack[top++] = k; \
        stack[top++] = m; \
        stack[top++] = d; \
        stack[top++] = num_open; \
    } while (0)
    #define pop(k, m, d, num_open) do { \
        num_open = stack[--top]; \
        d = stack[--top]; \
        m = stack[--top]; \
        k = stack[--top]; \
    } while (0)
    // every solution starts out by opening a below-pair at position 0
    push(0, -1, 0, 0);
    while (top) {
        pop(k, m, d, num_open);
        int64_t* openings = open + 2 * k + 2;
        openings[0] = openings[-2];
        openings[1] = openings[-1];
        int32_t avail = availability[k];
        // On CPU, this macro trick improves perf over 10% by letting the compiler
        // take advantage of the fact that d can only be 0 or 1.
        // Makes no difference on GPU.
        #define place_macro(d) do { \
            if (m>=0) { \
                pos[m] = k; \
                avail ^= (lsb32 << m); \
                openings[d] &= (openings[d] - 1); \
            } else { \
                openings[d] |= (nn1 >> k); \
                ++num_open; \
            } \
        } while (0)
        if (d) {
            place_macro(1);
        } else {
            place_macro(0);
        }
        ++k;
        availability[k] = avail;
        if (k == two_n) {
            // this is equivalent to results.push_back(pos);
            // p_results[0] is a counter;  after it follow the data
            // atomic increment of counter in device memory (i.e., RAM DIMMs on the GPU board)
            int64_t cnt = atomicAdd((unsigned long long*)p_result, (unsigned long long)1);
            if (cnt < kLimit) {
                constexpr int kAlignedCnt = (n + 7) / 8;
                int64_t* dst = p_result + 1 + (kAlignedCnt * cnt);
                #pragma unroll
                for (int i=0; i<kAlignedCnt; ++i) {
                    dst[i] = pgpualigned[i];
                }
            }
            // if cnt reaches or exceeds kLimit, that will be detected and the program will fail
        } else {
            // A super-naive way to divide the work across threads.  A hash of the current state at k_limit
            // determines whether the current thread should be pursuing a completion from this state or not.
            // The depth k_limit is chosen empirically to be both shallow enough so it's quick to reach and
            // deep enough to allow plenty of concurrency. This seems to work remarkably well in practice.
            constexpr int8_t k_limit = (n > 19 ? (8 + (n / 3)) : (n - 5));
            if (kNumLogicalThreads > 1 &&
                k == k_limit &&
                // multiply by a nice Mersenne prime to divide the work evenly across the threads... it works well...
                uint64_t(131071 * (openings[1] - openings[0]) + avail) % kNumLogicalThreads != logical_thread_index) {
                // some other thread will work on this
                continue;
            }
            // Now push on the stack the the children of the current node in the search tree.
            int8_t offset = k - two_n - 2;
            for (d=0; d<2; ++d) {
                if (openings[d]) { // if there is an opening, try closing it
                    m = offset + __ffsll(openings[d]);
                    // m could be -1, for example if the decision at pos k-1 was to open;
                    // only m from 0 .. n - 1 are useful
                    if (((unsigned)m < n) && ((avail >> m) & 1)) {
                        if (m || k <= n) { // this dedups L <==> R reversal twins
                            push(k, m, d, num_open);
                        }
                    }
                }
            }
            if (num_open < n) {
                push(k, -1, 1, num_open);
                push(k, -1, 0, num_open);
            }
        }
    }
}

template <int n>
__global__
void dfs_gpu(int64_t* p_result, const int32_t num_devices, const int32_t device_index) {
    __shared__ Availability<n> availability[kThreadsPerBlock];
    // PositionsGPU<n> pos;
    __shared__ Open<n> open[kThreadsPerBlock];
    // there are 2*n positions with 3 decisions per position and 4 bytes per decision on the stack
    __shared__ Stack<n> stack[kThreadsPerBlock];
    __shared__ PositionsGPUAligned<n> pgpualigned[kThreadsPerBlock];
    // the size of the arrays above add up to ~2KB for n=32
    // this bodes well for fitting tousands of threads inside on-chip memory
    // assume 1D grid of 1D blocks of threads
    const int32_t result_index = blockIdx.x * kThreadsPerBlock + threadIdx.x;
    const int32_t logical_thread_index =  result_index * num_devices + device_index;
    dfs<n>(p_result,
           availability[threadIdx.x],
           open[threadIdx.x],
           stack[threadIdx.x],
           pgpualigned[threadIdx.x],
           logical_thread_index,
           0);
}

// Sort the vector of solution sequences and count the unique ones.
// Optionally print each unique one.
template <int n>
int64_t unique_count(Results<n> &results) {
    int64_t total = results.size();
    int64_t unique = total;
    sort(results.begin(), results.end());
    if (kPrint && total) {
        print<n>(results[0]);
    }
    for (int i=1; i<total; ++i) {
        if (results[i] == results[i-1]) {
            --unique;
        } else if (kPrint) {
            print<n>(results[i]);
        }
    }
    return unique;
}


// ----------------------------- crux of solution ends here ------------------------------------------
// The rest is boring utilities for pretty printing, argument parsing, validation, gpu management, etc
// ---------------------------------------------------------------------------------------------------

// Return number of milliseconds elapsed since Jan 1, 1970 00:00 GMT.
long unixtime() {
    // There is the unix way, the navy way, and the C++11 way... apparently.
    using namespace chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    // Using steady_clock instead of system_clock above produces comically incorrect results.
    // For some reason, this also produces laughable results on Microsoft Windows PCs.
}

void cdo(const char* txt, cudaError_t err) {
    if (err == cudaSuccess) {
        return;
    }
    cout << unixtime() << " Failed to " << txt << " (Error code " << cudaGetErrorString(err) << ")\n" << flush;
    exit(-1);
}

// Start and manage the computation on GPU device "device"
template <int n>
void run_gpu_d(mutex& mtx, int& running, int64_t* count, Results<n>& final_results,
               int num_devices, int device) {
    int64_t* results_device;
    int64_t* results_host;
    auto t0 = unixtime();
    cudaEvent_t start, stop;
    cdo("choose device",
        cudaSetDevice(device));
    cdo("create start timer event",
        cudaEventCreate(&start));
    cdo("create stop timer event",
        cudaEventCreate(&stop, cudaEventBlockingSync));
    assert(sizeof(int64_t) == 8);
    constexpr int64_t kAlignedCnt = (n + 7) / 8;
    cdo("allocate GPU memory",
        cudaMalloc(&results_device, (1 + kLimit * kAlignedCnt) * sizeof(int64_t)));
    cdo("copy results[0] values from host to GPU",
        cudaMemcpy(results_device, count, sizeof(int64_t), cudaMemcpyHostToDevice));  // only results[0] is copied
    mtx.lock();
    {
        cout << unixtime() << " GPU " << device << " init took "
             << (unixtime() - t0) / 1000.0 << " seconds.\n" << flush;
    }
    mtx.unlock();
    auto t1 = unixtime();
    float milliseconds = 0;
    cdo("insert start event",
        cudaEventRecord(start));
    // -------------------------------------
    int blocks_x = div_up(kNumLogicalThreads, kThreadsPerBlock * num_devices);
    dim3 blocks(blocks_x);
    dfs_gpu<n><<<blocks, kThreadsPerBlock>>>(results_device, num_devices, device);
    // --------------------------------------
    cdo("DFS kernel launch",
        cudaPeekAtLastError());
    cdo("insert stop event",
        cudaEventRecord(stop));
    cdo("wait for GPU computation to complete",
        cudaEventSynchronize(stop));
    cdo("compute GPU elapsed time",
        cudaEventElapsedTime(&milliseconds, start, stop));
    mtx.lock();
    {
        cout << unixtime() << " GPU " << device << " computation took " << milliseconds/1000.0
             << " seconds on GPU clock, " << (unixtime() - t1)/1000.0 << " seconds on host clock.\n" << flush;
    }
    mtx.unlock();
    // first copy the count
    cdo("copy count from GPU to host",
        cudaMemcpy(count, results_device, sizeof(int64_t), cudaMemcpyDeviceToHost));
    if (*count >= kLimit) {
        cout << "Result for n = " << n << " will be bogus because GPU "
             << device << " exceeded " << kLimit << " solutions.\n" << flush;
    }
    int64_t r_count = *count;
    int64_t data_size = (1 + r_count * kAlignedCnt) * sizeof(int64_t);
    cdo("allocate host memory for result data",
        cudaMallocHost(&results_host, data_size));
    cdo("copy results from GPU to host",
        cudaMemcpy(results_host, results_device, data_size, cudaMemcpyDeviceToHost));
    // Compact results into a vector
    mtx.lock();
    for (int i=0; i<r_count; ++i) {
        Positions<n> pos;
        PositionsGPU<n>& gpos = *((PositionsGPU<n>*)(results_host + 1 + (kAlignedCnt * i)));
        for (int j=0; j<n; ++j) {
            pos[j] = gpos[j];
        }
        final_results.push_back(pos);
    }
    mtx.unlock();
    // todo: copy the entries
    cdo("free GPU memory for result",
        cudaFree(results_device));
    cdo("free temp host memory",
        cudaFreeHost(results_host));
    mtx.lock();
    --running;
    mtx.unlock();
}

// Start a CPU thread to manage each GPU device and wait for the computation to end.
template <int n>
void run_gpu(const int64_t* known_results) {
    int64_t* count;
    int64_t total;
    mutex mtx;
    Results<n> final_results;
    {
        cout << "\n";
        cout << "------\n";
        cout << unixtime() << " Computing PL(2, " << n << ")\n" << flush;
    }
    if (n > kMaxN) {
        cout << unixtime() << " Sorry, n = " << n << " exceeds the max allowed " << kMaxN << "\n" << flush;
        return;
    }
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    if (num_devices == 0) {
        cout << unixtime() << " It appears this system does not have any CUDA-capable GPUs.\n";
        return;
    }
    if (num_devices > kMaxDevices) {
        cout << unixtime() << " This program will use only " << kMaxDevices << " of this system's ";
        cout << num_devices << " CUDA-capable GPUs.\n";
        num_devices = kMaxDevices;
    } else {
        cout << unixtime() << " Will use " << num_devices << " system GPU(s).\n";
    }
    for (int device=0;  device<num_devices; ++device) {
        cdo("allocate host memory",
            cudaMallocHost(&count, num_devices * sizeof(int64_t)));
    }
    for (int device=0;  device<num_devices;  ++device) {
        count[device] = 0;
    }
    // Start a CPU thread for each GPU device.
    int running = num_devices;
    for (int device=0;  device < num_devices;  ++device) {
        auto runit = [&](int dev) {
            run_gpu_d<n>(mtx, running, count + dev, final_results, num_devices, dev);
        };
        thread(runit, device).detach();
    }
    // Wait for the treads to finish.
    bool devices_are_working = true;
    while (devices_are_working) {
        mtx.lock();
        devices_are_working = (running > 0);
        mtx.unlock();
        if (devices_are_working) {
            this_thread::sleep_for(chrono::milliseconds(50));
        }
    }
    // Sort and unique count on CPU.
    total = unique_count<n>(final_results);
    cdo("free host memory for counts",
        cudaFreeHost(count));
    {
        cout << unixtime() << " Result " << total << " for n = " << n;
        if (n < 0 || n >= 64 || known_results[n] == -1) {
            cout << " is NEW";
        } else if (known_results[n] == total) {
            cout << " MATCHES previously published result";
        } else {
            cout << " MISMATCHES previously published result " << known_results[n];
        }
        cout << "\n------\n\n";
    }
}

void init_known_results(int64_t (&known_results)[64]) {
    for (int i=0;  i<64; ++i) {
        known_results[i] = 0;
    }
    // There are no published results for n >= 29
    for (int i = 29;  i<64;  ++i) {
        if (i % 4 == 3 || i % 4 == 0) {
            known_results[i] = -1;
        }
    }
    known_results[3]  = 1;
    known_results[4]  = 0;
    known_results[7]  = 0;
    known_results[8]  = 4;
    known_results[11] = 16;
    known_results[12] = 40;
    known_results[15] = 194;
    known_results[16] = 274;
    known_results[19] = 2384;
    known_results[20] = 4719;
    known_results[23] = 31856;
    known_results[24] = 62124;
    known_results[27] = 426502;
    known_results[28] = 817717;
}

template <int n>
void print(const Positions<n>& pos) {
    cout << unixtime() << " Sequence ";
    int s[2 * n];
    for (int i=0; i<2*n; ++i) {
        s[i] = -1;
    }
    for (int m=1;  m<=n;  ++m) {
        int k2 = pos[m-1];
        int k1 = k2 - m - 1;
        assert(0 <= k1);
        assert(k2 < 2*n);
        assert(s[k1] == -1);
        assert(s[k2] == -1);
        s[k1] = s[k2] = m;
    }
    for (int i=0;  i<2*n;  ++i) {
        const int64_t m = s[i];
        assert(0 <= m);
        assert(m <= n);
        cout << setw(3) << m;
    }
    cout << "\n";
}


int main(int argc, char **argv) {
    int64_t known_results[64];
    init_known_results(known_results);

    if (argc != 2) 
    {
        printf("Specify the levels (i.e., the number of gpus)\n");
        exit(1);
    }

    const int level = atoi(argv[1]);

    switch (level)
    {
        case 1: run_gpu<12>(known_results); break;
        case 2: run_gpu<15>(known_results); break;
        case 4: run_gpu<19>(known_results); break;
        case 8: run_gpu<20>(known_results); break;
        default: return 0;
    }

    /* we cannot do 3 and 4 anymore due to unrolling
    run_gpu<3>(known_results);
    run_gpu<4>(known_results);
     */
    //run_gpu<7>(known_results);
    //run_gpu<8>(known_results);
    //run_gpu<11>(known_results);
    //run_gpu<12>(known_results);
    //run_gpu<15>(known_results);
    //run_gpu<16>(known_results);
    //run_gpu<19>(known_results);
    //run_gpu<20>(known_results);
    //run_gpu<23>(known_results);
    //run_gpu<24>(known_results);
    //run_gpu<27>(known_results);
    //run_gpu<28>(known_results);
    return 0;
}
