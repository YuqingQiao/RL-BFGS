#include <iostream>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include <cmath>
#include <chrono>
#include <string>
#include <iomanip>
#include "../include/forward_kinematics.hpp"
#include "../include/capsule_distance.hpp"
#include "../include/cma_es.hpp"

//Todo For debugging in python
template <typename MatrixType>
void printVariableDetails(const char* varName, const MatrixType& mat) {
    if constexpr (std::is_base_of<Eigen::MatrixBase<MatrixType>, MatrixType>::value) {
        std::cout << "Variable: " << varName << std::endl;
        std::cout << "Type: " << typeid(typename MatrixType::Scalar).name() << std::endl;
        std::cout << "Rows: " << mat.rows() << std::endl;
        std::cout << "Cols: " << mat.cols() << std::endl;
        std::cout << "Value:\n" << mat << std::endl;
    } else if constexpr (std::is_same_v<MatrixType, std::vector<Eigen::VectorXf>>) {
        std::cout << "Variable: " << varName << std::endl;
        std::cout << "Type: std::vector<Eigen::VectorXf>" << std::endl;
        std::cout << "Size: " << mat.size() << std::endl;
        std::cout << "Values:\n";
        for (const auto& vec : mat) {
            std::cout << vec.transpose() << std::endl;  // Transpose for horizontal display
        }
    } else if constexpr (std::is_same_v<MatrixType, std::vector<Eigen::Matrix4f>>) {
        std::cout << "Variable: " << varName << std::endl;
        std::cout << "Type: std::vector<Eigen::Matrix4f>" << std::endl;
        std::cout << "Size: " << mat.size() << std::endl;
        std::cout << "Values:\n";
        for (const auto& matrix : mat) {
            std::cout << matrix << "\n\n";
        }
    } else if constexpr (std::is_same_v<MatrixType, std::vector<std::pair<Eigen::VectorXf, float>>>) {
        std::cout << "Variable: " << varName << std::endl;
        std::cout << "Type: std::vector<std::pair<Eigen::VectorXf, float>>" << std::endl;
        std::cout << "Size: " << mat.size() << std::endl;
        std::cout << "Values:\n";
        for (const auto& p : mat) {
            std::cout << "Vector: " << p.first.transpose() << ", Value: " << p.second << std::endl;
        }
    }
}

// Overload for float types
void printVariableDetails(const char* varName, float value) {
    std::cout << "Variable: " << varName << std::endl;
    std::cout << "Type: float" << std::endl;
    std::cout << "Value: " << value << std::endl;
}

// Overload for int types
void printVariableDetails(const char* varName, int value) {
    std::cout << "Variable: " << varName << std::endl;
    std::cout << "Type: int" << std::endl;
    std::cout << "Value: " << value << std::endl;
}

#define PRINT(VAR) printVariableDetails(#VAR, VAR)

void print_values(const std::vector<Obstacle>& obstacles_vector, const std::pair<std::vector<Capsule>, std::vector<Capsule>>& capsules) {
    // Print obstacles
    for(const auto& obstacle : obstacles_vector) {
        std::cout << "Obstacle Position: [" << obstacle.pos[0] << ", " << obstacle.pos[1] << ", " << obstacle.pos[2] << "] Size: [" << obstacle.size[0] << ", " << obstacle.size[1] << ", " << obstacle.size[2] << "]\n";
    }

    // Print capsules from the first vector
    for(const auto& capsule : capsules.first) {
        std::cout << "Capsule p: [" << capsule.p[0] << ", " << capsule.p[1] << ", " << capsule.p[2] << "] u: [" << capsule.u[0] << ", " << capsule.u[1] << ", " << capsule.u[2] << "] roh: " << capsule.roh << "\n";
    }

    // Print capsules from the second vector
    for(const auto& capsule : capsules.second) {
        std::cout << "Capsule p: [" << capsule.p[0] << ", " << capsule.p[1] << ", " << capsule.p[2] << "] u: [" << capsule.u[0] << ", " << capsule.u[1] << ", " << capsule.u[2] << "] roh: " << capsule.roh << "\n";
    }
}

static std::chrono::high_resolution_clock::time_point init;
static std::chrono::high_resolution_clock::time_point generate;
static std::chrono::high_resolution_clock::time_point evaluate;
static std::chrono::high_resolution_clock::time_point ofun;
static std::chrono::high_resolution_clock::time_point update;
static std::chrono::high_resolution_clock::time_point sort;
static std::chrono::high_resolution_clock::time_point resample;
static std::chrono::high_resolution_clock::time_point total;


// Accumulated durations
static double generate_duration = 0;
static double evaluate_duration = 0;
static double update_duration = 0;
static double sort_duration = 0;
static double resample_duration = 0;
static double total_duration = 0;
static double ofun_duration = 0;

void tic(std::chrono::high_resolution_clock::time_point& time_point) {
    time_point = std::chrono::high_resolution_clock::now();
}

void toc(const std::chrono::high_resolution_clock::time_point& time_point) {
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = stop - time_point;

    if (&time_point == &generate) {
        generate_duration += duration.count();
    } else if (&time_point == &evaluate) {
        evaluate_duration += duration.count();
    } else if (&time_point == &update) {
        update_duration += duration.count();
    } else if (&time_point == &sort) {
        sort_duration += duration.count();
    } else if (&time_point == &resample) {
        resample_duration += duration.count();
    } else if (&time_point == &total) {
        total_duration += duration.count();
    } else if (&time_point == &ofun) {
        ofun_duration += duration.count();
    }
}

void print() {
    std::cout << "--------------\n";
    std::cout << std::fixed << std::setprecision(10);

    std::cout << "generate: \t" << generate_duration << "\n";
    std::cout << "evaluate: \t" << evaluate_duration << "\n";
    std::cout << "objfun: \t" << ofun_duration << "\n";
    std::cout << "update: \t" << update_duration << "\n";
    std::cout << "sorted: \t" << sort_duration << "\n";
    std::cout << "resample: \t" << resample_duration << "\n";
    std::cout << "ctotal: \t" << total_duration << "\n";
}

struct Options {
    float robot_base[3];
    float alpha;
    float beta;
    float gamma;
    float sigma;
    int ngen;
    int popsize;
    float ftol;
    float bounds[2][7];
};


extern "C" {

    typedef struct {
        Eigen::VectorXf current_q;
        Obstacle obstacles[8];
        Eigen::Vector3f robot_base;
        Eigen::Vector3f target_pos;
        Eigen::Matrix3f target_rot;
        float alpha, beta, gamma, sigma, ftol;
        int ngen, popsize;
        std::pair<Eigen::VectorXf, Eigen::VectorXf> bounds;
    } IKContext;

    float obj_fun(const Eigen::VectorXf& q, const IKContext& context);
    float obj_fun_p(const Eigen::VectorXf& q, const IKContext& context);
    float calc_dist(Eigen::VectorXf& q, IKContext& context);
    std::pair<Eigen::VectorXf , float> cma_es(IKContext context);

    void solve_ik(const float q[7], const Obstacle obstacles[8], const float target_pos[3], const float target_rot_arr[3], const Options* options, float q_res[7], float* f_res) {
        generate_duration = 0;
        evaluate_duration = 0;
        update_duration = 0;
        sort_duration = 0;
        resample_duration = 0;
        total_duration = 0;
        ofun_duration = 0;

        // assign context
        IKContext context;
        context.current_q = Eigen::Map<const Eigen::VectorXf>(q, 7);
        for(int i = 0; i < 8; i++) {
            context.obstacles[i] = obstacles[i];
        }
        context.target_pos = Eigen::Map<const Eigen::Vector3f>(target_pos);
        context.robot_base = Eigen::Map<const Eigen::Vector3f>(options->robot_base);
        context.target_rot = Eigen::AngleAxisf(target_rot_arr[0], Eigen::Vector3f::UnitX())
                           * Eigen::AngleAxisf(target_rot_arr[1]+1*M_PI, Eigen::Vector3f::UnitY())
                           * Eigen::AngleAxisf(target_rot_arr[2]+1*M_PI, Eigen::Vector3f::UnitZ());
        context.alpha = options->alpha;
        context.beta = options->beta;
        context.gamma = options->gamma;
        context.sigma = options->sigma;
        context.ngen = options->ngen;
        context.popsize = options->popsize;
        context.ftol = options->ftol;

        Eigen::VectorXf lower_bounds(7);
        Eigen::VectorXf upper_bounds(7);
        for (int i = 0; i < 7; i++) {
            lower_bounds[i] = options->bounds[0][i];
            upper_bounds[i] = options->bounds[1][i];
        }
        context.bounds = std::make_pair(lower_bounds, upper_bounds);

        // Apply CMA-ES
        std::pair<Eigen::VectorXf, float> res = cma_es(context);

        for (int i = 0; i < 7; i++) {
            q_res[i] = res.first[i];
        }
        *f_res = res.second;
    }

    std::pair<Eigen::VectorXf , float> cma_es(IKContext context) {

        CMA cma(context.current_q, context.sigma, context.popsize, context.bounds);

        // pre-allocate memory
        std::vector<Eigen::VectorXf> individuals;
        individuals.reserve(context.popsize);

        std::vector<std::pair<Eigen::VectorXf, float>> current_population;
        current_population.reserve(context.popsize);

        std::vector<std::pair<Eigen::VectorXf, float>> cached_pop;
        cached_pop.reserve(context.ngen * context.popsize);

        for (int gen = 0; gen < context.ngen; gen++) {

            // Generate a new population
            individuals = cma.generate();

            // Evaluate the individuals
            current_population.clear();
            for (const auto& ind : individuals) {
                float fitness = obj_fun(ind, context);
                current_population.push_back({ind, fitness});
                cached_pop.push_back({ind, fitness});
            }

            // Calculate the range of fitness values
            auto minmax = std::minmax_element(current_population.begin(), current_population.end(),
                [](const std::pair<Eigen::VectorXf, float>& a, const std::pair<Eigen::VectorXf, float>& b) {
                    return a.second < b.second;
                });
            float frange = minmax.second->second - minmax.first->second;;

            // Stop if frange is below threshold
            if (frange < context.ftol) {
//                std::cout << gen << "\n";
                break;
            }

            // Update the strategy with the evaluated individuals
            cma.update(current_population);
        }
        // Sort the cached_pop based on fitness
        std::sort(
            cached_pop.begin(),
            cached_pop.end(),
            [](const std::pair<Eigen::VectorXf, float>& a, const std::pair<Eigen::VectorXf, float>& b) {
                      return a.second < b.second;
                  }
        );
        // Now resample until we have a feasible solution
        std::pair<Eigen::VectorXf, float> res;
        float dist;
        //normal search
//        for (auto& [ind, fit] : cached_pop) {
//            res.first = ind;
//            res.second = fit;
//            dist = calc_dist(ind, context);
//            if (dist > 0) {
//                break;
//            }
//        }
        //skip search by steps
        int step = 5;
        for (int i = 0;;i = i + step) {
            res.first = cached_pop[i].first;
            res.second = cached_pop[i].second;
            dist = calc_dist(cached_pop[i].first, context);
//            step = step - 1;
            if (dist > 0) {
                break;
            }
//            if (step < 2) {
//                step = 2;
//            }
        }
//        std::cout << "-----------------------------------------------\n";
//        std::cout << "Resample:" << i << "\n";
//        std::cout << "C-Dist: " << dist << "\n";
        float test = obj_fun_p(res.first, context);
        return res;
    }

    float calc_dist(Eigen::VectorXf& q, IKContext& context) {
        // forward kinematics
        std::vector<Eigen::Matrix4f> fk_list = forward_kinematics(context.robot_base, q);
        // capsule positions
        std::vector<Obstacle> obstacles_vector(std::begin(context.obstacles), std::end(context.obstacles));
        std::pair<std::vector<Capsule>, std::vector<Capsule>> capsules = get_capsule_pos(fk_list, obstacles_vector);
        // min distance
        float min_dist = calc_min_distance(capsules);

        return min_dist;
    }

    float obj_fun(const Eigen::VectorXf& q, const IKContext& context) {
        std::vector<Eigen::Matrix4f> fk_list = forward_kinematics(context.robot_base, q);

        Eigen::Vector3f current_pos = fk_list.back().block<3,1>(0,3);
        Eigen::Matrix3f current_rot = fk_list.back().block<3,3>(0,0);

        float pos_error = (current_pos - context.target_pos).norm();
        float orientation_error = ((current_rot.transpose() * context.target_rot).trace()-1.0f) * 0.5f;
        orientation_error = acosf(std::max(-1.0f, std::min(1.0f, orientation_error)));
        float angle_error = (q - context.current_q).norm();

        float error = context.alpha * pos_error + context.beta * orientation_error + context.gamma * angle_error;
        return error;
    }

    float obj_fun_p(const Eigen::VectorXf& q, const IKContext& context) {
        std::vector<Eigen::Matrix4f> fk_list = forward_kinematics(context.robot_base, q);

        Eigen::Vector3f current_pos = fk_list.back().block<3,1>(0,3);
        Eigen::Matrix3f current_rot = fk_list.back().block<3,3>(0,0);

        float pos_error = (current_pos - context.target_pos).norm();
        float orientation_error = ((current_rot.transpose() * context.target_rot).trace()-1.0f) * 0.5f;
        orientation_error = acosf(std::max(-1.0f, std::min(1.0f, orientation_error)));
        float angle_error = (q - context.current_q).norm();

//        PRINT(current_pos);
//        PRINT(current_rot);
//
//        std::cout << "Pos: " << pos_error << std::endl;
//        std::cout << "Orient: " << orientation_error << std::endl;
//        std::cout << "Angle: " << angle_error << std::endl;

        float error = context.alpha * pos_error + context.beta * orientation_error + context.gamma * angle_error;
        return error;
    }
}