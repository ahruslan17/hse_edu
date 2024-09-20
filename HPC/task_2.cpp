#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <memory>


void dgemm_core(int M, int N, int K, const double* A, const double* B, double* C) {
    std::memset(C, 0, M * N * sizeof(double));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

void parallel_dgemm(int M, int N, int K, const double* A, const double* B, double* C, int num_procs) {
    omp_set_num_threads(num_procs);
    dgemm_core(M, N, K, A, B, C);
}

void init_random_matrix(int M, int N, double* A) {
    std::srand(static_cast<unsigned>(std::time(0)));
    for (int i = 0; i < M * N; ++i) {
        A[i] = std::rand() % 100;
    }
}

int main() {
    const std::vector<int> thread_counts = {1, 2, 4, 8, 16};
    const std::vector<int> sizes = {500, 1000, 1500};
    
    std::cout<<"Threads,Size,Time,GFLOPS\n";

    for (const int size : sizes) {
        const int M = size;
        const int N = size;
        const int K = size;

        double *A = new double[M * K];
        double *B = new double[K * N];
        double *C = new double[M * N];


        init_random_matrix(M, K, A);
        init_random_matrix(K, N, B);

        for (const int num_procs : thread_counts) {
            
            double start = omp_get_wtime();
            parallel_dgemm(M, N, K, A, B, C, num_procs);
            double end = omp_get_wtime();
            double time_taken = end - start;

            // GFLOPS = 2 * M * N * K / time_taken / 1e9
            double gflops = 2.0 * M * N * K / (time_taken * 1e9);

            std::cout<<num_procs<<","<<size<<","<<time_taken<<","<<gflops<<"\n";
        }
        
        delete[] A;
        delete[] B;
        delete[] C;
    }

    return 0;
}
