#include <iostream>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include <cmath>
#include <chrono>
#include <string>
#include <iomanip>
#include "../include/cppoptlib/meta.h"
#include "../include/cppoptlib/problem.h"
#include "../include/cppoptlib/solver/bfgssolver.h"
#include "../include/cppoptlib/solver/lbfgssolver.h"
#include "../include/cppoptlib/solver/gradientdescentsolver.h"
#include "../include/cppoptlib/solver/newtondescentsolver.h"
#include "../include/forward_kinematics.hpp"
#include "../include/capsule_distance.hpp"
#include "../include/cma_es.hpp"
#include <bits/stdc++.h>

using namespace cppoptlib;
using Eigen::VectorXd;

struct Options {
    float robot_base[3];
    float alpha;
    float beta;
    float gamma;
    float fDelta;
    float xDelta;
    float gradNorm;
    int maxiter;
    int population;
    float sigma;
    int skip;
    float bounds[2][7];
};


extern "C" {

    typedef struct {
        Eigen::VectorXf current_q;
        Obstacle obstacles[8];
        Eigen::Vector3f robot_base;
        Eigen::Vector3f target_pos;
        Eigen::Matrix3f target_rot;
        float alpha, beta, gamma, xDelta, fDelta, sigma, gradNorm;
        int maxiter, population, skip;
        std::pair<Eigen::VectorXf, Eigen::VectorXf> bounds;
    } IKContext;

    IKContext context;

    float obj_fun(const Eigen::VectorXf& q, const IKContext& context);
    float calc_dist(Eigen::VectorXf& q, IKContext& context);

    std::vector<std::pair<Eigen::VectorXf, float>> cached_pop;
    float callback_error;

    class obj : public Problem<float>{
    public:

        using typename cppoptlib::Problem<float>::Scalar;
        using typename cppoptlib::Problem<float>::TVector;

        float value(const Eigen::VectorXf &q) {
            std::vector<Eigen::Matrix4f> fk_list = forward_kinematics(context.robot_base, q);

            Eigen::Vector3f current_pos = fk_list.back().block<3,1>(0,3);
            Eigen::Matrix3f current_rot = fk_list.back().block<3,3>(0,0);

            float pos_error = (current_pos - context.target_pos).norm();
            float orientation_error = ((current_rot.transpose() * context.target_rot).trace()-1.0f) * 0.5f;
            orientation_error = acosf(std::max(-1.0f, std::min(1.0f, orientation_error)));
            float angle_error = (q - context.current_q).norm();

            float error = context.alpha * pos_error + context.beta * orientation_error + context.gamma * angle_error;
            callback_error = error;
            return error;
        }

        bool callback(const cppoptlib::Criteria<Scalar> &state, const Eigen::VectorXf &q) {

            cached_pop.push_back({q, callback_error});
            return true;
        }

    };

    void solve_ik(const float q[7], const Obstacle obstacles[8], const float target_pos[3], const float target_rot_arr[3], const Options* options, float q_res[7], float* f_res) {

        // assign context
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
        context.xDelta = options->xDelta;
        context.fDelta = options->fDelta;
        context.maxiter = options->maxiter;
        context.sigma = options->sigma;
        context.gradNorm = options->gradNorm;
        context.population = options->population;
        context.skip = options->skip;

        Eigen::VectorXf lower_bounds(7);
        Eigen::VectorXf upper_bounds(7);
        for (int i = 0; i < 7; i++) {
            lower_bounds[i] = options->bounds[0][i];
            upper_bounds[i] = options->bounds[1][i];
        }
        context.bounds = std::make_pair(lower_bounds, upper_bounds);

        cached_pop;

        //gaussian generation
        for (int i = 0; i < context.population;i = i + 1) {
            float res_g[7];
            float fitness_gg;
            for (int j = 0; j < 7;j = j + 1) {
                std::mt19937 generator(std::random_device{}());
                std::normal_distribution<float> distribution(q[j] ,context.sigma);
                res_g[j] = {distribution(generator)};
            }
            Eigen::VectorXf res_gg;
            res_gg = Eigen::Map<const Eigen::VectorXf>(res_g, 7);
            fitness_gg = obj_fun(res_gg, context);
            cached_pop.push_back({res_gg, fitness_gg});
        }
        //sort
        std::sort(
            cached_pop.begin(),
            cached_pop.end(),
            [](const std::pair<Eigen::VectorXf, float>& a, const std::pair<Eigen::VectorXf, float>& b) {
                      return a.second < b.second;
                  }
        );


        // Apply bfgs
        obj f;
        Eigen::VectorXf x;
        x = cached_pop[0].first;//context.current_q;cached_pop[0].first
        BfgsSolver<obj> solver;
        // user config bfgs
        obj::TCriteria crit = obj::TCriteria::defaults();
        crit.iterations = context.maxiter;
        crit.xDelta = context.xDelta;
        crit.fDelta = context.fDelta;
        crit.gradNorm = context.gradNorm;
        solver.setStopCriteria(crit);

        solver.minimize(f, x);
        //save q and fitness
//        Eigen::VectorXf res0 = x;
//        float fitness = f(x);
//        cached_pop.push_back({res0, fitness});

        std::pair<Eigen::VectorXf, float> res;
        //sort
        std::sort(
            cached_pop.begin(),
            cached_pop.end(),
            [](const std::pair<Eigen::VectorXf, float>& a, const std::pair<Eigen::VectorXf, float>& b) {
                      return a.second < b.second;
                  }
        );
        //collision check
        float dist;
        for (int i = 0;;i = i + context.skip) {
            res.first = cached_pop[i].first;
            res.second = cached_pop[i].second;
            dist = calc_dist(cached_pop[i].first, context);
            if (dist > 0) {
//                std::cout<<cached_pop.size()<<std::endl;
                break;
            }
        }
        cached_pop.clear();
        for (int i = 0; i < 7; i++) {
            q_res[i] = res.first[i];
        }
        *f_res = res.second;
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

}