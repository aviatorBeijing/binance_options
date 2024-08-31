#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <boost/math/distributions/students_t.hpp>

// Function to read historical prices from a CSV file
std::vector<double> read_prices_from_csv(const std::string& filename) {
    std::vector<double> prices;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        prices.push_back(std::stod(line));
    }
    return prices;
}

// Function to calculate log returns
std::vector<double> calculate_log_returns(const std::vector<double>& prices) {
    std::vector<double> log_returns;
    for (size_t i = 1; i < prices.size(); ++i) {
        log_returns.push_back(std::log(prices[i] / prices[i-1]));
    }
    return log_returns;
}

// Negative log-likelihood function for MLE using Student's t-distribution
double negative_log_likelihood(const std::vector<double>& log_returns, double mu, double sigma, double nu) {
    double n = log_returns.size();
    double log_likelihood = 0.0;

    boost::math::students_t dist(nu);

    for (double r : log_returns) {
        double z = (r - mu) / sigma;
        log_likelihood += std::log(boost::math::pdf(dist, z) / sigma);
    }

    return -log_likelihood; // Return negative log-likelihood
}

// Function to find the MLE estimates for mu, sigma, and nu using a basic grid search
std::tuple<double, double, double> find_mle(const std::vector<double>& log_returns) {
    double best_mu = 0.0, best_sigma = 1.0, best_nu = 3.0;
    double best_nll = std::numeric_limits<double>::max(); // Best negative log-likelihood

    // Grid search parameters
    double mu_start = -0.1, mu_end = 0.1, mu_step = 0.001;
    double sigma_start = 0.01, sigma_end = 0.5, sigma_step = 0.001;
    double nu_start = 2.0, nu_end = 10.0, nu_step = 0.1;

    for (double mu = mu_start; mu <= mu_end; mu += mu_step) {
        for (double sigma = sigma_start; sigma <= sigma_end; sigma += sigma_step) {
            for (double nu = nu_start; nu <= nu_end; nu += nu_step) {
                double nll = negative_log_likelihood(log_returns, mu, sigma, nu);
                if (nll < best_nll) {
                    best_nll = nll;
                    best_mu = mu;
                    best_sigma = sigma;
                    best_nu = nu;
                }
            }
        }
    }

    return {best_mu, best_sigma, best_nu};
}

int main() {
    std::vector<double> prices = read_prices_from_csv("aapl_prices.csv");
    std::vector<double> log_returns = calculate_log_returns(prices);
    auto [mu_mle, sigma_mle, nu_mle] = find_mle(log_returns);

    std::cout << "Estimated mu: " << mu_mle << std::endl;
    std::cout << "Estimated sigma: " << sigma_mle << std::endl;
    std::cout << "Estimated nu (degrees of freedom): " << nu_mle << std::endl;

    return 0;
}
    

