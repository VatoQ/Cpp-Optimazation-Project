#include <cstddef>
#include <iostream>
#include <random>
#include <chrono>
//#include <algorithm>
#include <vector>
#include <cmath>
#include <iomanip>

//#define CHECK_RESULTS
#define ALTERNATIVE_LOOP


typedef float real;
constexpr size_t N = 1 << 14; // 14 is probably the sweet spot
                              //
constexpr size_t VECTOR_WIDTH_BYTES = 32;
constexpr size_t BATCH_SIZE = 45; //VECTOR_WIDTH_BYTES / sizeof(real);
constexpr size_t RUNS = 8;
constexpr size_t FREE_RUNS = 2;
constexpr size_t ALIGNMENT = 32;

constexpr real EPS = 1e-6;

void calculate_average(
        real* matrix, 
        real* vector_batch, 
        real* result_batch,
        real* ref_result_batch,
        size_t N,
        size_t runs,
        double glops_total);


int main()
{

    real* matrix = static_cast<real*>(
            std::aligned_alloc(ALIGNMENT, N * N * sizeof(real))
            );
     
    real* vector_batch = static_cast<real*>(
            std::aligned_alloc(ALIGNMENT, N * BATCH_SIZE * sizeof(real))
            );

    real* result_batch = static_cast<real*>(
            std::aligned_alloc(ALIGNMENT, N * BATCH_SIZE * sizeof(real))
            );


    real* ref_result_batch = static_cast<real*>(
            std::aligned_alloc(ALIGNMENT, N * BATCH_SIZE * sizeof(real))
            );


    const size_t FLOPS = 2ULL * BATCH_SIZE * N * N;
    const double GFLOPS = FLOPS * 1.0 / 1'000'000'000;
    
    std::cout << "Initiating benchmark:" <<
        "\n\tProblem Size: " << N << " entries" <<
        "\n\tMatrix Size: " << N * N << " entries" <<
        "\n\tBatch Size: " << N * BATCH_SIZE <<
        "\n\tTotal Data in Matrix: " << 
        N * N * sizeof(real) * 1.0 / ( 1024 * 1024 * 1024) << " GiB" <<
        "\n\tTotal Data in Batch: " << 
        N * BATCH_SIZE * sizeof(real) * 1.0 / (1024 * 1024) << " MiB\n\n";

    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);

    for (size_t i = 0; i < N * N; i++)
    {
        matrix[i] = dist(e2);
    }

    for (size_t i = 0; i < N * BATCH_SIZE; i++)
    {
        vector_batch[i] = dist(e2);
    }
    
    calculate_average(
            matrix, 
            vector_batch, 
            result_batch, 
            ref_result_batch, 
            N, 
            RUNS, 
            GFLOPS);

    std::free(matrix);
    std::free(vector_batch);
    std::free(result_batch);
    std::free(ref_result_batch);
}


inline void dot_product(
        real* sums, 
        real* matrix, 
        real* vector_batch, 
        size_t n, 
        size_t N, 
        size_t k)
{
    for (size_t b = 0; b < BATCH_SIZE; b++)
    {
        sums[b] += matrix[n * N + k] * vector_batch[b * N + k];
    }
}
#pragma omp declare simd notinbranch aligned(matrix, vector_batch, sums : ALIGNMENT)
inline void dot_product_vectorized(
        real* sums, 
        real* matrix, 
        real* vector_batch, 
        size_t n, 
        size_t N, 
        size_t k)
{
// #pragma omp simd aligned(matrix, vector_batch, sums : ALIGNMENT)
    for (size_t b = 0; b < BATCH_SIZE; b++)
    {
        sums[b] += matrix[n * N + k] * vector_batch[b * N + k];
    }
}

void batch_mult_opt(
        real* __restrict matrix, 
        real* __restrict vector_batch, 
        real* __restrict result_batch, 
        size_t N)
{
#ifndef ALTERNATIVE_LOOP

    #pragma omp parallel for schedule(static)
    for (size_t n = 0; n < N; n++)
    {
        alignas(ALIGNMENT) real sums[BATCH_SIZE] = {0.f};
        for (size_t k = 0; k < N; k++)
        {
            dot_product_vectorized(sums, matrix, vector_batch, n, N, k);
        }

        for (size_t i = 0; i < BATCH_SIZE; i++)
        {
            result_batch[i * N + n] = sums[i];
        }
    }
#else
    #pragma omp parallel for schedule(static)
    for (size_t n = 0; n < N; n++)
    {
        for (size_t b = 0; b < BATCH_SIZE; b++)
        {
            real sum = 0.f;

            #pragma omp simd aligned(matrix, vector_batch : ALIGNMENT)
            for (size_t k = 0; k <N; k++)
            {
                sum += matrix[n * N + k] * vector_batch[b * N + k];
            }

            result_batch[b * N + n] = sum;
        }
    }

#endif // ALTERNATIVE_LOOP
}

void batch_mult_ref(
        real* __restrict matrix, 
        real* __restrict vector_batch, 
        real* __restrict result_batch, 
        size_t N)
{
    for (size_t n = 0; n < N; n++)
    {
        real sums[BATCH_SIZE] = {0.f};
        for (size_t k = 0; k < N; k++)
        {
            dot_product(sums, matrix, vector_batch, n, N, k);
        }

        for (size_t i = 0; i < BATCH_SIZE; i++)
        {
            result_batch[i * N + n] = sums[i];
        }
    }
}


real check_results(
        real* matrix,
        real* vector_batch,
        real* result_batch,
        real* ref_result_batch,
        size_t N)
{
    batch_mult_ref(matrix, vector_batch, ref_result_batch, N);

    real max_error = 0.f;

    for (size_t i = 0; i < N * BATCH_SIZE; i++)
    {
        real error = std::abs(ref_result_batch[i] - result_batch[i]);

        if (error > max_error)
        {
            max_error = error;
        }
    }
    return max_error;
}

double run(
        real* matrix,
        real* vector_batch,
        real* result_batch,
        real* ref_result_batch,
        size_t N,
        size_t run_id)
{
    auto start = std::chrono::high_resolution_clock::now();
    batch_mult_opt(matrix, vector_batch, result_batch, N);
    auto end = std::chrono::high_resolution_clock::now();
    auto seconds_obj = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start);


    double seconds = seconds_obj.count() * 1e-6;


    #ifdef CHECK_RESULTS
    real diff = check_results(
            matrix, 
            vector_batch, 
            result_batch, 
            ref_result_batch, 
            N);

    if (diff > EPS)
    {
        real ref_max = *std::max_element(
                ref_result_batch, 
                ref_result_batch + N);

        
        real ref_min = *std::min_element(
                ref_result_batch, 
                ref_result_batch + N);

        real ref_range = std::max(std::abs(ref_max), std::abs(ref_min));

        real relative_error = diff / (ref_range + 1e-6f) * 100.f;

        if (relative_error > 0.001)
        {
            std::cout << "\tINCORRECT RESULTS: magnitude of error: " 
                << diff << ", relative error: " 
                << relative_error << " %\n";
        }
    }


    #endif

    std::cout << std::fixed << std::setprecision(5)
        << "Duration: " << seconds << " s. in run " << run_id;

    return seconds;

}


void calculate_average(
        real* matrix, 
        real* vector_batch, 
        real* result_batch,
        real* ref_result_batch,
        size_t N,
        size_t runs,
        double gflops_total)
{
    //float sum = 0.;
    //float gflop_sum = 0.;
    std::vector<double> runtimes;
    std::vector<double> gflops_values;

    for (size_t run_id = 0; run_id < runs; run_id++)
    {
        double runtime = run(
                matrix, 
                vector_batch, 
                result_batch, 
                ref_result_batch, 
                N, 
                run_id);

        double gflops = gflops_total / runtime;

        std::cout << std::fixed << std::setprecision(5)
            << " - GFLOPS: " << gflops;

        if (run_id > FREE_RUNS - 1)
        {
            //sum += runtime;
            //gflop_sum += gflops;
            runtimes.push_back(runtime);
            gflops_values.push_back(gflops);
            std::cout << "\n";
        }
        else {
            std::cout << " - not in Average\n";
        }


    }

    double sum_runtime = 0.0;
    for (double t : runtimes) sum_runtime += t;

    float avg = sum_runtime / (runs - FREE_RUNS);

    double gflop_sum = 0.0;
    for (double g : gflops_values) gflop_sum += g;
    float gflop_avg = gflop_sum / (runs - FREE_RUNS);

    double variance_runtime = 0.0;

    for (double t : runtimes)
    {
        double diff = t - avg;
        variance_runtime += diff * diff;
    }

    variance_runtime /= (runtimes.size() -1);
    double stddev_runtime = std::sqrt(variance_runtime);


    double variance_gflops = 0.0;

    for (double g : gflops_values)
    {
        double diff = g - gflop_avg;
        variance_gflops += diff * diff;
    }

    variance_gflops /= (gflops_values.size() -1);
    double stddev_gflops = std::sqrt(variance_gflops);


    std::cout << std::fixed << std::setprecision(5)
        << "Average Runtime: " 
        << avg << " ± " << stddev_runtime << ", performance: " 
        << gflop_avg << " ± " << stddev_gflops << " GFLOPS" 
        << std::endl;
}







