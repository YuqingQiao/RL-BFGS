#pragma once

#include <Eigen/Dense>
#include <vector>

extern "C" {
    std::vector<Eigen::Matrix4f> forward_kinematics(const Eigen::Vector3f& T_wb, const Eigen::VectorXf& q);
}
