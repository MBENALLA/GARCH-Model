#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <numeric>
#include <cmath>
#include <random>
#include <Eigen/Dense>
#include <utility>

#include "NelderMeadOptimizer.h"

/*
lecture d'un fichier CSV contenant des prix de clôture et
calcule des variations en pourcentage entre ces prix.
*/ 
std::vector<double> calculatePctChange(const std::string& filename) {
    std::ifstream file(filename);

    if (!file) {
        std::cout << "Erreur lors de l'ouverture du fichier." << std::endl;
        return std::vector<double>();
    }

    std::vector<double> closePrices;
    std::string line;

    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        std::getline(ss, value, ',');
        std::getline(ss, value, ',');

        double closePrice = std::stod(value);
        closePrices.push_back(closePrice);
    }

    file.close();

    std::vector<double> pctChange;
    for (size_t i = 1; i < closePrices.size(); ++i) {
        double change = (closePrices[i] - closePrices[i - 1]) / closePrices[i - 1];
        double pctChangeValue = change * 100.0;
        pctChange.push_back(pctChangeValue);
    }

    return pctChange;
}

/*
Génération d'un certain nombre de rendements aléatoires
à partir d'une distribution normale.
*/  
std::vector<double> generateRandomReturns(int numReturns) {
    std::vector<double> returns;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> distribution(0.0, 1.0); 

    for (int i = 0; i < numReturns; ++i) {
        double value = distribution(gen);
        returns.push_back(value);
    }

    return returns;
}

/*
Calcul de la volatilite de long terme en utilisant
les variables omega, alpha et beta (Process GARCH)
*/ 
double longRun(double omega, double alpha, double beta) {
    double long_run = std::sqrt(omega / (1 - alpha - beta));
    return long_run;
}

/*
Calcul de la volatilité réalisée à partir
d'un ensemble de rendements et d'une valeur moyenne mu (Process GARCH)
*/ 
std::pair<std::vector<double>, std::vector<double>> realizedVolatility(const std::vector<double>& returns, double mu) {
    size_t size = returns.size();
    std::vector<double> residuals(size);
    std::vector<double> realized_volatility(size);

    for (size_t t = 0; t < size; t++) {
        residuals[t] = returns[t] - mu;
        realized_volatility[t] = std::abs(residuals[t]);
    }

    return std::make_pair(residuals, realized_volatility);
}

/*
Calcul de la volatilité conditionnelle à partir d'un ensemble 
de résidus et de paramètres requis par un process GARCH
*/ 
std::vector<double> conditionalVolatility(const std::vector<double>& residuals, double long_run, double omega, double alpha, double beta) {
    size_t size = residuals.size();
    std::vector<double> conditional_volatility(size);

    conditional_volatility[0] = long_run;

    for (size_t t = 1; t < size; t++) {
        conditional_volatility[t] = std::sqrt(omega + alpha * std::pow(residuals[t-1], 2) + beta * std::pow(conditional_volatility[t-1], 2));
    }

    return conditional_volatility;
}

// Estimation du maximum de vraisemblance (MLE) pour un process GARCH.
double garch_mle(Eigen::Matrix<double, 4, 1> params) {
    double mu = params[0];
    double omega = params[1];
    double alpha = params[2];
    double beta = params[3];

    std::vector<double> returns = generateRandomReturns(10);
    int size = returns.size();

    double long_run = longRun(omega, alpha, beta);

    std::pair<std::vector<double>, std::vector<double>> res = realizedVolatility(returns, mu);

    std::vector<double> residuals = res.first;
    std::vector<double> realized_volatility = res.second;

    std::vector<double> conditional_volatility = conditionalVolatility(residuals, long_run, omega, alpha, beta); 

    std::vector<double> likelihood(returns.size());
    double log_likelihood = 0.0;

    for (size_t t = 0; t < returns.size(); t++) {
        likelihood[t] = 1 / (std::sqrt(2 * M_PI) * conditional_volatility[t]) * std::exp(-std::pow(realized_volatility[t], 2) / (2 * std::pow(conditional_volatility[t], 2)));
        log_likelihood += std::log(likelihood[t]);
    }

    return -log_likelihood;
}

double calculate_variance(const std::vector<double>& returns, double mean) {
    double variance = 0.0;
    for (double value : returns) {
        variance += pow(value - mean, 2);
    }
    variance /= returns.size();
    return variance;
}

double calculateRMSE(const std::vector<double>& predictions, const std::vector<double>& targets) {
    size_t size = predictions.size();

    if (size != targets.size()) {
        throw std::runtime_error("Les vecteurs de prédictions et de cibles doivent avoir la même taille.");
    }

    double sumSquaredErrors = 0.0;

    for (size_t i = 0; i < size; ++i) {
        double error = predictions[i] - targets[i];
        sumSquaredErrors += error * error;
    }

    double meanSquaredError = sumSquaredErrors / size;
    double rmse = std::sqrt(meanSquaredError);

    return rmse;
}

std::vector<double> simulateMonteCarlo(double mu, double sigma, int numSimulations) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<double> distribution(mu, sigma);

    std::vector<double> simulations(numSimulations);

    for (int i = 0; i < numSimulations; ++i) {
        double sample = distribution(generator);
        simulations[i] = sample;
    }

    return simulations;
}

int main() {
    /*
    valeurs de clôture pour->
    Actif: ETHEREUM/BUSD
    Produit: Futures Perputual Contract
    Interval = 1min
    Periode = 31/06/23 - 06/07/23
    */  

    std::cout << "***********" << std::endl;
    std::cout << "   Modelisation de Volatilité-->" << std::endl;
    std::cout << "   Actif: ETHEREUM/BUSD" << std::endl;
    std::cout << "   Produit: Futures Perpetual Contract" << std::endl;
    std::cout << "   Interval = 1min" << std::endl;
    std::cout << "   Periode = 31/06/23 - 06/07/23" << std::endl;
    std::cout << "***********\n" << std::endl;

    std::string filename = "closes_ETHBUSD_PERP.csv";
    std::vector<double> pctChange = calculatePctChange(filename);    
    // std::vector<double> pctChange = generateRandomReturns(10);

    double mean = std::accumulate(pctChange.begin(), pctChange.end(), 0.0) / pctChange.size();
    double variance = calculate_variance(pctChange, mean);

    Eigen::Matrix<double, 4, 1> start(mean, variance, 0, 0);

    std::cout << "Paramètres initiaux : " << std::endl;
    std::cout << "\tmean: " << mean << std::endl;
    std::cout << "\tvariance: " << variance << "\n" << std::endl;
    
    auto result = Nelder_Mead_Optimizer<4>(garch_mle, start, 0.1, 10e-10);
    double mu = result[0];
    double omega = result[1];
    double alpha = result[2];
    double beta = result[3];

    double long_run  = longRun(omega, alpha, beta);

    std::pair<std::vector<double>, std::vector<double>> res = realizedVolatility(pctChange, mu);
    std::vector<double> residuals = res.first;
    std::vector<double> realized_volatility = res.second;

    std::vector<double> conditional_volatility = conditionalVolatility(residuals, long_run, omega, alpha, beta); 

    double ll = garch_mle(start);

    std::cout << "\nParamètres optimaux : " << std::endl;
    std::cout << "\tmu: " << mu << std::endl;
    std::cout << "\tomega: " << omega << std::endl;
    std::cout << "\talpha: " << alpha << std::endl;
    std::cout << "\tbeta: " << beta << std::endl;
    std::cout << "\tlong-run volatility: " << long_run << std::endl;
    // std::cout << "\tmle: " << ll << std::endl;

    double rmse = calculateRMSE(conditional_volatility, realized_volatility);
    std::cout << "\tRMSE: " << rmse << std::endl;

    double sigma = std::sqrt(omega);
    int numSimulations = pctChange.size();

    std::vector<double> monteCarloSimulations = simulateMonteCarlo(mu, sigma, numSimulations);
    double rmse_mc = calculateRMSE(monteCarloSimulations, pctChange);

    std::cout << "\nSimulation Monte-Carlo (MC) des rendements : " << std::endl;
    std::cout << "\tSimulation MC (RMSE): " << rmse_mc << std::endl;

    return 0;
}