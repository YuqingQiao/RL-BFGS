#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <typeinfo>
#include <numeric>

#include "../include/cma_es.hpp"

////Todo For debugging in python
//
//template <typename MatrixType>
//void printVariableDetails(const char* varName, const MatrixType& mat) {
//    if constexpr (std::is_base_of<Eigen::MatrixBase<MatrixType>, MatrixType>::value) {
//        std::cout << "Variable: " << varName << std::endl;
//        std::cout << "Type: " << typeid(typename MatrixType::Scalar).name() << std::endl;
//        std::cout << "Rows: " << mat.rows() << std::endl;
//        std::cout << "Cols: " << mat.cols() << std::endl;
//        std::cout << "Value:\n" << mat << std::endl;
//    } else if constexpr (std::is_same_v<MatrixType, std::vector<Eigen::VectorXf>>) {
//        std::cout << "Variable: " << varName << std::endl;
//        std::cout << "Type: std::vector<Eigen::VectorXf>" << std::endl;
//        std::cout << "Size: " << mat.size() << std::endl;
//        std::cout << "Values:\n";
//        for (const auto& vec : mat) {
//            std::cout << vec.transpose() << std::endl;  // Transpose for horizontal display
//        }
//    } else if constexpr (std::is_same_v<MatrixType, std::vector<std::pair<Eigen::VectorXf, float>>>) {
//        std::cout << "Variable: " << varName << std::endl;
//        std::cout << "Type: std::vector<std::pair<Eigen::VectorXf, float>>" << std::endl;
//        std::cout << "Size: " << mat.size() << std::endl;
//        std::cout << "Values:\n";
//        for (const auto& p : mat) {
//            std::cout << "Vector: " << p.first.transpose() << ", Value: " << p.second << std::endl;
//        }
//    }
//}
//
//// Overload for float types
//void printVariableDetails(const char* varName, float value) {
//    std::cout << "Variable: " << varName << std::endl;
//    std::cout << "Type: float" << std::endl;
//    std::cout << "Value: " << value << std::endl;
//}
//
//// Overload for int types
//void printVariableDetails(const char* varName, int value) {
//    std::cout << "Variable: " << varName << std::endl;
//    std::cout << "Type: int" << std::endl;
//    std::cout << "Value: " << value << std::endl;
//}
//
//#define PRINT(VAR) printVariableDetails(#VAR, VAR)



CMA::CMA(Eigen::VectorXf centroid_init, float sigma_init, int lambda_init, std::pair<Eigen::VectorXf, Eigen::VectorXf> bounds_init)
        : centroid(centroid_init), sigma(sigma_init), lambda_(lambda_init), bounds(bounds_init) {

        dim = centroid.size();
        pc = Eigen::VectorXf::Zero(dim);
        ps = Eigen::VectorXf::Zero(dim);
        chiN = std::sqrt(dim) * (1 - 1.f / (4.f * dim) + 1.f / (21.f * dim * dim));
        C = Eigen::MatrixXf::Identity(dim, dim);

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(C);
        diagD = Eigen::VectorXf::Ones(dim);
        B = Eigen::MatrixXf::Identity(dim, dim);

        BD = B * diagD.asDiagonal();

        update_count = 0;
        computeParams();
}


std::vector<Eigen::VectorXf> CMA::generate() {
    Eigen::MatrixXf arz = Eigen::MatrixXf::Random(lambda_, dim);
    std::vector<Eigen::VectorXf> population(lambda_);

    Eigen::MatrixXf add = arz * BD.transpose() * sigma;

    for (int i = 0; i < lambda_; i++) {
        population[i] = (add.row(i).transpose() + centroid)
                        .cwiseMax(bounds.first)
                        .cwiseMin(bounds.second);
    }
    return population;
}

void CMA::update(std::vector<std::pair<Eigen::VectorXf, float>>& population){
    std::sort(population.begin(), population.end(),
              [](const std::pair<Eigen::VectorXf, float>& a, const std::pair<Eigen::VectorXf, float>& b) {
                  return a.second < b.second;
              });

    Eigen::MatrixXf individuals(lambda_, dim);
    for (int i = 0; i < lambda_; i++) {
        individuals.row(i) = population[i].first;
    }

    Eigen::VectorXf old_centroid = centroid;
    centroid = weights.transpose() * individuals.block(0, 0, mu, dim);
    Eigen::VectorXf c_diff = centroid - old_centroid;

    ps = (1 - cs) * ps + std::sqrt(cs * (2 - cs) * mueff) / sigma * B * diagD.cwiseInverse().asDiagonal() * B.transpose() * c_diff;
    float hsig = (ps.norm() / std::sqrt(1.f - std::pow(1.f - cs, 2.f * (update_count + 1))) / chiN < (1.4f + 2.f / (dim + 1.f))) ? 1.f : 0.f;
    update_count++;

    pc = (1 - cc) * pc + hsig * std::sqrt(cc * (2 - cc) * mueff) / sigma * c_diff;
    Eigen::MatrixXf artmp = individuals.block(0, 0, mu, dim).rowwise() - old_centroid.transpose();
    C = (1 - ccov1 - ccovmu + (1 - hsig) * ccov1 * cc * (2 - cc)) * C + ccov1 * pc * pc.transpose() + ccovmu / std::pow(sigma, 2) * artmp.transpose() * weights.asDiagonal() * artmp;

    sigma *= std::exp((ps.norm() / chiN - 1.) * cs / damps);

    // sometimes the values can overflow, so we add a small regularization to C
    float epsilon = 1e-10;
    C = C + epsilon * Eigen::MatrixXf::Identity(dim, dim);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(C);

    diagD = es.eigenvalues().cwiseSqrt();
    B = es.eigenvectors();
    BD.noalias() = B * diagD.asDiagonal();
}

void CMA::computeParams() {

    mu = lambda_ / 2;

    weights = Eigen::VectorXf(mu);

    for (int i = 0; i < mu; i++) {
        weights(i) = std::log(mu + 0.5f) - std::log(i + 1.f);
    }

    weights /= weights.sum();

    mueff = 1.f / weights.squaredNorm();

    cc = 4.f / (dim + 4.f);
    cs = (mueff + 2.f) / (dim + mueff + 3.f);
    ccov1 = 2.f / (std::pow(dim + 1.3f, 2) + mueff);

    ccovmu = 2.f * (mueff - 2.f + 1.f / mueff) / (std::pow(dim + 2.f, 2) + mueff);
    ccovmu = std::min(1.f - ccov1, ccovmu);
    damps = 1.f + 2.f * std::max(0.f, std::sqrt((mueff - 1.f) / (dim + 1.f)) - 1.f) + cs;
}