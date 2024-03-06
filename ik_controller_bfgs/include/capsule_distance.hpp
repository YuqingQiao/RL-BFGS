#pragma once

#include <Eigen/Dense>

// Declare the Capsule struct
struct Capsule {
    Eigen::Vector3f p;
    Eigen::Vector3f u;
    float roh;
};

struct Obstacle {
    float pos[3];
    float size[3];
};

extern "C"{
    float calc_min_distance(
        const std::pair<std::vector<Capsule>, std::vector<Capsule>>& capsules_pair
    );
    std::pair<std::vector<Capsule>, std::vector<Capsule>> get_capsule_pos(
        const std::vector<Eigen::Matrix4f>& fk_list,
        const std::vector<Obstacle>& obstacles
    );
}
