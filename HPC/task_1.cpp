#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <cmath>

void dgemm_core(int M, int N, int K, const double* A, const double* B, double* C, bool parallel) {
    std::memset(C, 0, M * N * sizeof(double));

    if (parallel) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < K; ++k) {
                    C[i * N + j] += A[i * K + k] * B[k * N + j];
                }
            }
        }
    } else {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < K; ++k) {
                    C[i * N + j] += A[i * K + k] * B[k * N + j];
                }
            }
        }
    }
}

void sequential_dgemm(int M, int N, int K, const double* A, const double* B, double* C) {
    dgemm_core(M, N, K, A, B, C, false);
}

void parallel_dgemm(int M, int N, int K, const double* A, const double* B, double* C) {
    dgemm_core(M, N, K, A, B, C, true);
}

bool compare_matrices(int size, const double* A, const double* B) {
    for (int i = 0; i < size; ++i) {
        if (std::fabs(A[i] - B[i]) > 1e-9) {
            return false;
        }
    }
    return true;
}

void init_random_matrix(int M, int N, double* A) {
    std::srand(static_cast<unsigned>(std::time(0)));
    for (int i=0; i < M * N; ++i) {
        A[i] = std::rand() % 100;
    }
}

int main() {
    const int m = 500;
    const int k = 300;
    const int n = 400;

    double *A = new double[m * k];
    double *B = new double[k * n];
    double *C_seq = new double[m *n];
    double *C_par = new double[m * n];

    init_random_matrix(m,k,A);
    init_random_matrix(k,n,B);

    // последовательно
    double start = omp_get_wtime();
    sequential_dgemm(m, n, k, A, B, C_seq);
    double end = omp_get_wtime();
    std::cout << "Sequential DGEMM time: " << end - start << " seconds." << std::endl;

    // параллельно
    start = omp_get_wtime();
    parallel_dgemm(m, n, k, A, B, C_par);
    end = omp_get_wtime();
    std::cout << "Parallel DGEMM time: " << end - start << " seconds." << std::endl;

    // проверка корректности
    if (compare_matrices(m * n, C_seq, C_par)) {
        std::cout << "Results are identical." << std::endl;
    } else {
        std::cout << "Results differ!" << std::endl;
    }

    delete[] A;
    delete[] B;
    delete[] C_seq;
    delete[] C_par;

    return 0;
}
