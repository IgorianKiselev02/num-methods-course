#include <omp.h>
#include <vector>
#include <math.h>
#include "algos.hpp"

using namespace std;


// implemetation of block processing
void parallel_block(
    int BS,
    int N,
    int i,
    int j,
    vector<double> &dmvec,
    vector< vector<double> > &u,
    double (*f)(double, double)
) {
    for (int iblock = 0; iblock < BS; iblock++) {
        int ci = 1 + i * BS + iblock;

        if (ci <= N) {
            for (int jblock = 0; jblock < BS; jblock++) {
                int cj = 1 + j * BS + jblock;

                if (cj <= N) {
                    double temp = u[ci][cj];

                    u[ci][cj] = 0.25 * (u[ci - 1][cj] + u[ci + 1][cj] + u[ci][cj - 1] 
                        + u[ci][cj + 1] - f(1.0 * ci / (N + 1), 1.0 * cj / (N + 1)) / (N + 1) / (N + 1));

                    dmvec[ci] = max(dmvec[ci], fabs(temp - u[ci][cj]));
                }
            }
        }
    }
}


// implementation of algo 11.6
int parallel_algo(
    int N,
    int NB,
    int BS,
    double eps,
    int threads,
    vector< vector<double> > &u,
    double (*f)(double, double)
) {
    int result = 0;

    double dmax = 0.0;
    double h = 1.0 / (N + 1);

    omp_lock_t dmax_lock;
    omp_init_lock(&dmax_lock);

    do {
        vector<double> dmvec(N + 1, 0.0);

        dmax = 0.0;
        result++;
        // нарастание волны (размер волны равен nx+1)

        for (int nx = 0; nx < NB; nx++) {

            #pragma omp parallel for default(none) shared(N, BS, nx, u, f, dmvec) num_threads(threads)
            for (int i = 0; i < nx + 1; i++) {
                int j = nx - i;

                // <обработка блока с координатами (i,j)>
                parallel_block(BS, N, i, j, dmvec, u, f);
            } // конец параллельной области 
        }

        for (int nx = NB - 2; nx >= 0; nx--) {

            #pragma omp parallel for default(none) shared(N, BS, NB, nx, u, f, dmvec) num_threads(threads)
            for (int i = NB - nx - 1; i < NB; i++) {
                int j = 2 * NB - 2 - nx - i;

                // <обработка блока с координатами (i,j)>
                parallel_block(BS, N, i, j, dmvec, u, f);
            } // конец параллельной области 
        }

        // <определение погрешности вычислений> 

        int chunk = 200;

        #pragma omp parallel for default(none) shared(dmvec, dmax, N, chunk, dmax_lock) num_threads(threads)
        for (int i = 1; i < N + 1; i += chunk) {
            double d = 0.0;

            for (int j = i; j < i + chunk; j++) {
                if (d < dmvec[j]) {
                    d = dmvec[j]; 
                }
            }

            omp_set_lock(&dmax_lock);
            if (dmax < d) {
                dmax = d; 
            }
            omp_unset_lock(&dmax_lock);
        } // конец параллельной области
    } while (dmax > eps);

    return result;
}
