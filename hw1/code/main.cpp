#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <algorithm>

#include "algos.hpp"

using namespace std;


// pair of functions from example in the book
double f1(double x, double y) {
    return 0;
}


double g1(double x, double y) {
    double result = 0.0;

    if (x == 0) {
        result = 100 - 200 * y;
    } else if (x == 1) {
        result = 200 * y - 100;
    } else if (y == 0) {
        result = 100 - 200 * x;
    } else if (y == 1) {
        result = 200 * x - 100;
    }

    return result;
}


// "sleepy" test - let's find out how good parallel is for partly-slow functions
double f2(double x, double y) {
    if (x < 0.001 && y < 0.001) {
        this_thread::sleep_for(chrono::milliseconds(5));
    }
    return 0;
}


double g2(double x, double y) {
    double result = 0.0;

    if (x == 0) {
        result = 100 - 200 * y;
    } else if (x == 1) {
        result = 200 * y - 100;
    } else if (y == 0) {
        result = 100 - 200 * x;
    } else if (y == 1) {
        result = 200 * x - 100;
    }

    return result;
}


uniform_real_distribution<double> create_udistribution(int N, vector<vector<double>> &u, double (*fun)(double, double)) {
    vector<double> values;

    for (int i = 0; i < N + 2; i++) {
        // D borders are the same as in the example from book to simplify function calls
        double v1 = fun((double) i / (N + 1), 0);
        double v2 = fun(0, (double) i / (N + 1));
        double v3 = fun((double) i / (N + 1), 1);
        double v4 = fun(1, (double) i / (N + 1));

        u[i][0] = v1;
        u[0][i] = v2;
        u[i][N + 1] = v3;
        u[N + 1][i] = v4;

        values.push_back(v1);
        values.push_back(v2);
        values.push_back(v3);
        values.push_back(v4);
    }

    double lborder = *min_element(begin(values), end(values));
    double rborder = *max_element(begin(values), end(values));

    return uniform_real_distribution<double>(lborder, rborder);
}


vector<vector<double>> ugenerate(int N, double (*fun)(double, double)) {
    vector<vector<double>> u(N + 2, vector<double>(N + 2, 0));

    default_random_engine setup {};
    uniform_real_distribution<double> generator = create_udistribution(N, u, fun);

    // in book authors took random numbers for first approximation
    for (int i = 1; i < N + 1; i++) {
        for (int j = 1; j < N + 1; j++) {
            u[i][j] = generator(setup);
        }
    }

    return u;
}


// seems to be unused due to "sleepy" experiment
vector<vector<double>> fgenerate(int N, double (*fun)(double, double)) {
    vector<vector<double>> f(N + 2, vector<double>(N + 2, 0));

    for (int i = 0; i < N + 2; i++) {
        for (int j = 0; j < N + 2; j++) {
            f[i][j] = fun(i / (N + 1), j / (N + 1));
        }
    }

    return f;
}


void printer(vector<double> &values) {
    cout << "All launches sorted: ";

    for (int i = 0; i < values.size(); i++) {
        cout << values[i] << " ";
    }

    cout << ".\n";
}


void run_experiment(int N, int NB, int BS, double eps, double reps, double deviations) {
    vector<int> num_of_threads = {1, 2, 4, 8, 16};
    vector<double (*)(double, double)> f_functions = {f1, f2};
    vector<double (*)(double, double)>  g_functions = {g1, g2};

    cout << "\n\n\nExperiment begining.\n\n" << "Experiment using values:\n"  << "Grid size: " << N << ".\n"
        << "Number of blocks: " << NB << ".\n" << "Block size: " << BS << ".\n" << "Epsilon value: " << eps << ".\n"
        << "Runs per experiment: " << reps <<  ".\n" << "Deviations per experiment: " << deviations << ".\n" << "\n";

    // run unpareallel solution
    cout << "Unparallel launches started.\n";

    for (int func = 0; func < f_functions.size(); func++) {
        cout << "Func num: " << func << ".\n";
        vector<double> times;
        double avg_time = 0l;
        int result;

        for (int exp = 0; exp < reps; exp++) {
            cout << "Experiment num: " << exp << ".\n";
            std::vector<std::vector<double>> u = ugenerate(N, g_functions[func]);

            auto begin = chrono::high_resolution_clock::now();
            result = unparallel_algo(N, eps, u, f_functions[func]);
            auto end = chrono::high_resolution_clock::now();
            auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            times.push_back(1.0 * total_time / 1000000.0);
        }

        sort(times.begin(), times.end());

        for (int pos = deviations; pos < reps - deviations; pos++) {
            avg_time += times[pos];
        }

        printer(times);
        cout << "Unparallel solution average time: " << avg_time / (reps - 2 * deviations) << " seconds.\n";
        cout << "Iterations taken: " << result << ".\n\n";
    }

    cout << "Parallel launches started.\n";

    // run pareallel solution
    for (int i = 0; i < num_of_threads.size(); i++) {
        for (int func = 0; func < f_functions.size(); func++) {
            cout << "Func num: " << func << ".\n";

            vector<double> times;
            double avg_time = 0l;
            int result;

            for (int exp = 0; exp < reps; exp++) {
                cout << "Experiment num: " << exp << ".\n";

                std::vector<std::vector<double>> u = ugenerate(N, g_functions[func]);

                auto begin = chrono::high_resolution_clock::now();
                result = parallel_algo(N, NB, BS, eps, num_of_threads[i], u, f_functions[func]);
                auto end = chrono::high_resolution_clock::now();
                auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                times.push_back(1.0 * total_time / 1000000.0);
            }

            sort(times.begin(), times.end());

            for (int pos = deviations; pos < reps - deviations; pos++) {
                avg_time += times[pos];
            }

            printer(times);
            cout << "Parallel solution average time: " << 1.0 * avg_time / (reps - 2 * deviations) << " seconds.\n";
            cout << "Used threads: " << num_of_threads[i] << ".\n" << "Iterations taken: " << result << ".\n\n";
        }
    }
}


int main() {
    int N = 3000;
    int NB = 150;
    int BS = N / NB;
    double eps = 0.15;
    double reps = 20;
    double deviations = 2;

    run_experiment(N, NB, BS, eps, reps, deviations);

    return 0;
}
