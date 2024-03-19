#include <vector>
#include <math.h>
#include "algos.hpp"

using namespace std;


// implementation of algo 11.1
int unparallel_algo(
    int N,
    double eps,
    vector<vector<double>> &u,
    double (*f)(double, double)
) {
    int result = 0;
    double dmax = 0.0;
    double h = 1.0 / (N + 1);

    do {
        result++;
        dmax = 0;
        for (int i = 1; i < N + 1; i++) {
            for (int j = 1; j < N + 1; j++) {
                double temp = u[i][j];

                u[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] 
                    - h * h * f((double) i / (N + 1), (double) j / (N + 1))); 

                dmax = max(dmax, fabs(temp - u[i][j]));
            }
        }
    } while ( dmax > eps );
    
    return result;
}
