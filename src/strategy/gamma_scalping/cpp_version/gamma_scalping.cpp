#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>
#include <random>

namespace py = pybind11;

double black_scholes_call_price(double S, double K, double T, double r, double sigma) {
    double d1 = (std::log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*std::sqrt(T));
    double d2 = d1 - sigma*std::sqrt(T);
    double call_price = S * std::erfc(-d1/std::sqrt(2))/2 - K * std::exp(-r*T) * std::erfc(-d2/std::sqrt(2))/2;
    return call_price;
}

double black_scholes_delta(double S, double K, double T, double r, double sigma) {
    double d1 = (std::log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*std::sqrt(T));
    double delta = std::erfc(-d1/std::sqrt(2))/2;
    return delta;
}

double black_scholes_gamma(double S, double K, double T, double r, double sigma) {
    double d1 = (std::log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*std::sqrt(T));
    double gamma = std::exp(-0.5*d1*d1) / (S * sigma * std::sqrt(2*M_PI*T));
    return gamma;
}

py::array_t<double> simulate_gbm_paths(double S0, double r, double sigma, double T, double dt, int n_sim) {
    int N = static_cast<int>(T/dt);
    py::array_t<double> S_paths({n_sim, N});
    auto S_paths_mut = S_paths.mutable_unchecked<2>();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);

    for (int i = 0; i < n_sim; ++i) {
        S_paths_mut(i, 0) = S0;
        for (int j = 1; j < N; ++j) {
            double z = d(gen);
            S_paths_mut(i, j) = S_paths_mut(i, j-1) * std::exp((r - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * z);
        }
    }
    return S_paths;
}

py::array_t<double> gamma_scalping(py::array_t<double> S_paths, double K, double r, double sigma, double T, double dt) {
    auto S_paths_unchecked = S_paths.unchecked<2>();
    int n_sim = S_paths_unchecked.shape(0);
    int N = S_paths_unchecked.shape(1);

    py::array_t<double> pnl_array(n_sim);
    auto pnl = pnl_array.mutable_unchecked<1>();

    for (int i = 0; i < n_sim; ++i) {
        double cash = 0.0;
        double shares = 0.0;
        for (int t = 0; t < N; ++t) {
            double S = S_paths_unchecked(i, t);
            double tau = T - t * dt;
            double delta = (tau > 0) ? black_scholes_delta(S, K, tau, r, sigma) : 0.0;
            double gamma = (tau > 0) ? black_scholes_gamma(S, K, tau, r, sigma) : 0.0;

            if (t == 0) {
                shares = delta;
                cash = black_scholes_call_price(S, K, tau, r, sigma) - shares * S;
            } else {
                double shares_new = delta;
                double delta_shares = shares_new - shares;
                cash -= delta_shares * S;
                shares = shares_new;

                if (std::abs(S_paths_unchecked(i, t) - S_paths_unchecked(i, t-1)) > 0) {
                    double gamma_scalping_pnl = 0.5 * gamma * std::pow(S_paths_unchecked(i, t) - S_paths_unchecked(i, t-1), 2);
                    cash += gamma_scalping_pnl;
                }
            }
        }
        double payoff = std::max(S_paths_unchecked(i, N-1) - K, 0.0);
        pnl(i) = cash + shares * S_paths_unchecked(i, N-1) - payoff;
    }

    return pnl_array;
}

PYBIND11_MODULE(gamma_scalping, m) {
    m.def("simulate_gbm_paths", &simulate_gbm_paths, "Simulate GBM paths");
    m.def("gamma_scalping", &gamma_scalping, "Gamma scalping strategy");
}

