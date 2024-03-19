#ifndef ALGOS_HPP
#define ALGOS_HPP

#include <vector>

using namespace std;


int unparallel_algo(
    int N,
    double eps,
    vector<vector<double>> &u,
    double (*f)(double, double)
);


int parallel_algo(
    int N,
    int NB,
    int BS,
    double eps,
    int threads,
    vector<vector<double>> &u,
    double (*f)(double, double)
);


#endif // ALGOS_HPP