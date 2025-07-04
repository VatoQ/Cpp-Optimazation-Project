#include <iostream>
#include <random>
#include <chrono>
#include <algorithm>

//#define STRIP_MINING

constexpr unsigned long long N = 1 << 14;
constexpr float EPS = 1E-4;
constexpr bool CHECK_RESULTS = true;
constexpr unsigned long long RUNS = 5;
constexpr unsigned N_BLOCKS = 2; // For my Laptop, 1 seems the optimal amount of blocks with float precision
// for single precision it is 2
constexpr unsigned long long TILE = 64; //4 * N_BLOCKS;
constexpr size_t BLOCK_SIZE = 32;


void multiply(float* __restrict__ a, float* __restrict__ b, float* __restrict__ result, unsigned long long n)
{
    #ifndef STRIP_MINING   
    #pragma omp parallel for schedule(static)
	for (size_t i = 0; i < n; ++i)
    {
        float sum = 0.f;

        #pragma omp simd aligned(a, b:32)
        for (size_t j = 0; j < n; ++j)
        {
            sum += a[i * n + j] * b[j];
        }

        result[i] = sum;
    }

    
    #else
    # pragma omp parallel for schedule(static)
    for (unsigned long long i = 0; i < n; i++)
    {

        unsigned long long j = 0;
        float sum = 0.f;

        alignas(32) float sum_cache[TILE] = {0.f};
        for (; j + TILE <= n; j += TILE)
        {
            //#pragma GCC ivdep
            //#pragma omp simd aligned(sum_cache, a, b: 32)
            for (int k = 0; k < TILE; k++)
            {
                float individual_sum = sum_cache[k];
                float addend = a[i * n + j + k] * b[j + k];
                sum_cache[k] = individual_sum + addend;
                //sum_cache[k] += a[i * n + j + k] * b[j + k];
            }
        }

        for (; j < n; j++)
        {
            sum_cache[0] += a[i * n + j] * b[j];
        }

        for (int c = 0; c < TILE; c++)
        {
            sum += sum_cache[c];
        }

        result[i] = sum;
    }

    #endif //STRIP_MINING
}

void multiply_reference(float a[], float b[], float result[], unsigned long long n)
{
    for (unsigned long long i = 0; i < n; i++)
    {
        float sum = 0.f;
        for (unsigned long long j = 0; j < n; j++)
        {
            sum += a[i * n + j] * b[j];
        }
        result[i] = sum;
    }
}

float check_result(float a[], float b[], float result_opt[], float result_ref[], unsigned long long n)
{
    multiply_reference(a, b, result_ref, n);
    float max_error = 0.0f;
    for (unsigned long long i = 0; i < n; i++)
    {
        float error = std::abs(result_ref[i] - result_opt[i]);
        if (error > max_error)
        {
            max_error = error;
        }
    }
    return max_error;
}

float run(float a[], float b[], float result_opt[], float result_ref[], unsigned long long n, unsigned long long run_id)
{
    auto start = std::chrono::high_resolution_clock::now();
    multiply(a, b, result_opt, n);
    auto end = std::chrono::high_resolution_clock::now();
    auto seconds_obj = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    float seconds = seconds_obj.count() * 1e-6;
    if (CHECK_RESULTS)
    {
        float diff = check_result(a, b, result_opt, result_ref, n);
        if (diff > EPS)
        {
            float ref_max = *std::max_element(result_ref, result_ref + N);
            float ref_min = *std::min_element(result_ref, result_ref + N);
            float ref_range = std::max(std::abs(ref_max), std::abs(ref_min));
            float relative_error = diff / (ref_range + 1e-6f) * 100.0f;
            if (relative_error > 0.001)
            {
                std::cout << "\tINCORRECT RESULTS: magnitude of error: " << diff << ", relative error: " << relative_error << " %\n";
            }
        }
    }

    std::cout << "Duration: " << seconds << " s. in run " << run_id;

    return seconds;
}

void calculate_average(float a[], float b[], float result_opt[], float result_ref[], unsigned long long n, unsigned long long runs, float gflops_total)
{
    float sum = 0.;
    float gflop_sum = 0.;

    for (unsigned long long run_id = 0; run_id < runs; run_id++)
    {
        float runtime = run(a, b, result_opt, result_ref, n, run_id);

        // runtime is in microseconds. So, to get GFLOP per seconds divide by 1e-6
        float gflops = gflops_total / runtime;
        sum += runtime;
        gflop_sum += gflops;

        std::cout << " - GFLOPS: " << gflops << "\n";
    }

    float avg = sum / runs;
    float gflop_avg = gflop_sum / runs;

    std::cout << "Average Runtime: " << avg << ", performance: " << gflop_avg << " GFLOPS" << std::endl;
}

int main()
{
	//float a[N * N];
	//float b[N];
    //float result_opt[N];
    //float result_ref[N];

    const unsigned long long FLOPS = 2ULL * N * N;
    const float GFLOPS = FLOPS * 1.0 / 1'000'000'000;

    std::cout << "Starting benchmark for problem size " 
        << N << ". Matrix size: " << N * N 
        << "\nTotal data in matrix: " << (N * N) * sizeof(float) / (1024. * 1024.) 
        << " MiB. Total FLOP per run: " << FLOPS << "\n\n";
    // allocating aligned arrays
    float* a = static_cast<float*>(std::aligned_alloc(32, N * N * sizeof(float)));
    float* b = static_cast<float*>(std::aligned_alloc(32, N * sizeof(float)));
    float* result_opt = static_cast<float*>(std::aligned_alloc(32, N * sizeof(float)));
    float* result_ref = static_cast<float*>(std::aligned_alloc(32, N * sizeof(float)));





	
	std::random_device rd;

	std::mt19937 e2(rd());

	std::uniform_real_distribution<> dist(0, 1);

    for (unsigned long long i = 0; i < N * N; i++)
    {
        a[i] = dist(e2);
    }

	for (unsigned long long i = 0; i < N; i++)
	{
		b[i] = dist(e2);
	}


    calculate_average(a, b, result_opt, result_ref, N, RUNS, GFLOPS);

    std::free(a);
    std::free(b);
    std::free(result_opt);
    std::free(result_ref);
}
