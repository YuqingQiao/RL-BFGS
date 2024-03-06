#include <Eigen/Dense>
#include <vector>
#include <cmath>

inline Eigen::Matrix4f jointTransform1(float theta) {
    Eigen::Matrix4f transform;
    transform << std::cos(theta), -std::sin(theta), 0, 0,
                 std::sin(theta),  std::cos(theta), 0, 0,
                 0, 0, 1, 0.333,
                 0, 0, 0, 1;
    return transform;
}
inline Eigen::Matrix4f jointTransform2(float theta) {
    Eigen::Matrix4f transform;
    transform << std::cos(theta), -std::sin(theta), 0, 0,
                0, 0, 1, 0,
                -std::sin(theta), -std::cos(theta), 0, 0,
                0, 0, 0, 1;
    return transform;
}
inline Eigen::Matrix4f jointTransform3(float theta) {
    Eigen::Matrix4f transform;
    transform << std::cos(theta), -std::sin(theta), 0, 0,
                0, 0, -1, -0.316,
                std::sin(theta), std::cos(theta), 0, 0,
                0, 0, 0, 1;
    return transform;
}
inline Eigen::Matrix4f jointTransform4(float theta) {
    Eigen::Matrix4f transform;
    transform << std::cos(theta), -std::sin(theta), 0, 0.0825,
                0, 0, -1, 0,
                std::sin(theta), std::cos(theta), 0, 0,
                0, 0, 0, 1;
    return transform;
}
inline Eigen::Matrix4f jointTransform5(float theta) {
    Eigen::Matrix4f transform;
    transform << std::cos(theta), -std::sin(theta), 0, -0.0825,
                0, 0, 1, 0.384,
                -std::sin(theta), -std::cos(theta), 0, 0,
                0, 0, 0, 1;
    return transform;
}
inline Eigen::Matrix4f jointTransform6(float theta) {
    Eigen::Matrix4f transform;
    transform << std::cos(theta), -std::sin(theta), 0, 0,
                0, 0, -1, 0,
                std::sin(theta), std::cos(theta), 0, 0,
                0, 0, 0, 1;
    return transform;
}
inline Eigen::Matrix4f jointTransform7(float theta) {
    Eigen::Matrix4f transform;
    transform << std::cos(theta), -std::sin(theta), 0, 0.088,
                0, 0, -1, 0,
                std::sin(theta), std::cos(theta), 0, 0,
                0, 0, 0, 1;
    return transform;
}

inline Eigen::Matrix4f flangeTransform() {
    Eigen::Matrix4f transform;
    transform << 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0.207,
                0, 0, 0, 1;
    return transform;
}

extern "C" {

    std::vector<Eigen::Matrix4f> forward_kinematics(const Eigen::Vector3f& T_wb, const Eigen::VectorXf& q) {
        std::vector<Eigen::Matrix4f> fk_list;
        Eigen::Matrix4f fk = Eigen::Matrix4f::Identity();

        // Add robot base displacement
        fk.block<3,1>(0,3) += T_wb;
        fk_list.push_back(fk);

        // Directly apply each joint transform
        fk *= jointTransform1(q(0)); fk_list.push_back(fk);
        fk *= jointTransform2(q(1)); fk_list.push_back(fk);
        fk *= jointTransform3(q(2)); fk_list.push_back(fk);
        fk *= jointTransform4(q(3)); fk_list.push_back(fk);
        fk *= jointTransform5(q(4)); fk_list.push_back(fk);
        fk *= jointTransform6(q(5)); fk_list.push_back(fk);
        fk *= jointTransform7(q(6)); fk_list.push_back(fk);

        // For the flange without q
        fk *= flangeTransform(); fk_list.push_back(fk);

        return fk_list;
    }
}

