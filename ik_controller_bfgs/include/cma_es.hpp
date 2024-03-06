// CMA.hpp

#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>


class CMA {
private:
    Eigen::VectorXf centroid, pc, ps, diagD, weights;
    Eigen::MatrixXf B, C, BD;
    int dim, lambda_, mu, update_count;
    float sigma, chiN, mueff, cc, cs, ccov1, ccovmu, damps;
    std::pair<Eigen::VectorXf, Eigen::VectorXf> bounds;
    void computeParams();

public:
    CMA(Eigen::VectorXf centroid_init, float sigma_init, int lambda_init, std::pair<Eigen::VectorXf, Eigen::VectorXf> bounds_init);

    std::vector<Eigen::VectorXf> generate();
    void update(std::vector<std::pair<Eigen::VectorXf, float>>& population);
};

