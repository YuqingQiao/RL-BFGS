#include <iostream>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include <numeric>
#include "../include/capsule_distance.hpp"

Eigen::Vector2f closest_point_on_segment(const Eigen::Vector2f& a, const Eigen::Vector2f& b) {
    if (a.dot(b - a) > 0) return a;
    if (b.dot(a - b) > 0) return b;

    Eigen::Vector2f ab = b - a;
    float proj_scale = -a.dot(ab) / ab.dot(ab);
    return a + proj_scale * ab;
}

const Eigen::MatrixXf IDENTITY_3X3 = Eigen::MatrixXf::Identity(3, 3);

extern "C" {

//    float calc_min_distance(const std::pair<std::vector<Capsule>, std::vector<Capsule>>& capsules_pair) {
//
//    const std::vector<Capsule>& robot_capsules = capsules_pair.first;
//    const std::vector<Capsule>& obst_capsules = capsules_pair.second;
//
//    float min_dist = std::numeric_limits<float>::max();
//
//    std::vector<Eigen::Vector2f> vertices(4);
//
//    for (size_t i = 0; i < robot_capsules.size(); ++i) {
//        const Capsule& r_caps = robot_capsules[i];
//
//        for (size_t j = 0; j < obst_capsules.size(); ++j) {
//            const Capsule& o_caps = obst_capsules[j];
//                std::cout << "r_cap: " << i << " p: " << r_caps.p.transpose() << "\n";
//                std::cout << "o_cap: " << j << " p: " << o_caps.p.transpose() << "\n";
//
//            Eigen::Vector3f centerDist = o_caps.p - r_caps.p;
//            float estimatedMinDist = centerDist.norm() - (r_caps.u - r_caps.p).norm() - (o_caps.u - o_caps.p).norm();
//
//            if (estimatedMinDist > min_dist) {
//                continue; // Skip to the next obstacle capsule
//            }
//
//            Eigen::Vector3f s1 = r_caps.u - r_caps.p;
//            Eigen::Vector3f s2 = o_caps.u - o_caps.p;
//
//            Eigen::MatrixXf A(3,2);
//            A.col(0) = s2;
//            A.col(1) = -s1;
//
//            Eigen::Vector3f y = o_caps.p - r_caps.p;
//            Eigen::HouseholderQR<Eigen::MatrixXf> qr(A);
//            Eigen::MatrixXf basis = Eigen::MatrixXf::Identity(A.rows(), 2);
//            Eigen::MatrixXf Q = qr.householderQ() * basis;
//            Eigen::MatrixXf R = qr.matrixQR().topLeftCorner(2, 2).triangularView<Eigen::Upper>();
//
//            // 1. Print out the Q and R matrices from the QR factorization
////            std::cout << "Q matrix:\n" << Q << "\n";
////            std::cout << "R matrix:\n" << R << "\n";
//
//            auto u = [&](const Eigen::Vector2f& x) -> Eigen::Vector2f {
//                return R * x + Q.transpose() * y;
//            };
//
//            vertices[0] = u({0, 0});
//            vertices[1] = u({0, 1});
//            vertices[2] = u({1, 1});
//            vertices[3] = u({1, 0});
//
//            // 2. Print out all vertices
////            for (int v = 0; v < 4; ++v) {
////                std::cout << "Vertex " << v << ": " << vertices[v].transpose() << "\n";
////            }
//
//            int side_sum = 0;
//            Eigen::Vector2f u_min;
//            float min_distance = std::numeric_limits<float>::max();
//
//            for (int k = 0; k < 4; ++k) {
//                Eigen::Vector2f v1 = vertices[k];
//                Eigen::Vector2f v2 = (k != 3) ? vertices[k + 1] : vertices[0];
//
//                float res = -v1.y() * (v2.x() - v1.x()) + v1.x() * (v2.y() - v1.y());
//                side_sum += (res >= 0 ? 1 : -1);
//
//                Eigen::Vector2f temp = closest_point_on_segment(v1, v2);
//
//                // 3. Print out the "temp" variable from closest_point_on_segment
////                std::cout << "Temp point (for vertices " << k << " and " << (k != 3 ? k + 1 : 0) << "): " << temp.transpose() << "\n";
//
//                float dist = temp.norm();
//
//                // 4. Print out each calculated distance with the corresponding "i" and "j" value
////                std::cout << "Calculated distance (i=" << i << ", j=" << j << "): " << dist << "\n";
//
//                if (dist < min_distance) {
//                    min_distance = dist;
//                    u_min = temp;
//                }
//            }
//            if (std::abs(side_sum) == 4) {
//                u_min = Eigen::Vector2f(0, 0);
//            }
//            std::cout << "u_min: " << u_min.transpose() << "\n";
//            float dist = std::sqrt(u_min.squaredNorm() + y.squaredNorm() - y.dot(Q * Q.transpose() * y)) - r_caps.roh - o_caps.roh;
//            std::cout << "Min Dist: " << dist << "\n";
//            min_dist = std::min(min_dist, dist);
//
//        }
//    }
//    // 5. Print the resulting minimum distance
////    std::cout << "Final minimum distance: " << min_dist << "\n";
//
//    return min_dist;
//}

    float calc_min_distance(const std::pair<std::vector<Capsule>, std::vector<Capsule>>& capsules_pair) {

        const std::vector<Capsule>& robot_capsules = capsules_pair.first;
        const std::vector<Capsule>& obst_capsules = capsules_pair.second;

        float min_dist = std::numeric_limits<float>::max();

        std::vector<Eigen::Vector2f> vertices(4);

        for (size_t i = 0; i < robot_capsules.size(); ++i) {
            const Capsule& r_caps = robot_capsules[i];

            for (size_t j = 0; j < obst_capsules.size(); ++j) {
                const Capsule& o_caps = obst_capsules[j];

                // do a quick estimate of the minimum distance by using bounding spheres.
                // centers
                Eigen::Vector3f c_r = 0.5f * (r_caps.p + r_caps.u);
                Eigen::Vector3f c_o = 0.5f * (o_caps.p + o_caps.u);
                // half-lengths
                float h_r = (r_caps.p - r_caps.u).norm() + r_caps.roh;
                float h_o = (o_caps.p - o_caps.u).norm() + o_caps.roh;

                float estimatedMinDist = (c_r - c_o).norm() - h_r - h_o;

                if (estimatedMinDist > min_dist) {
                    continue; // Skip to the next capsule
                }

                Eigen::Vector3f s1 = r_caps.u - r_caps.p;
                Eigen::Vector3f s2 = o_caps.u - o_caps.p;

                Eigen::MatrixXf A(3,2);
                A.col(0) = s2;
                A.col(1) = -s1;

                Eigen::Vector3f y = o_caps.p - r_caps.p;
                Eigen::HouseholderQR<Eigen::MatrixXf> qr(A);
                Eigen::MatrixXf basis = Eigen::MatrixXf::Identity(A.rows(), 2);
                Eigen::MatrixXf Q = qr.householderQ() * basis;
                Eigen::MatrixXf R = qr.matrixQR().topLeftCorner(2, 2).triangularView<Eigen::Upper>();

                auto u = [&](const Eigen::Vector2f& x) -> Eigen::Vector2f {
                    return R * x + Q.transpose() * y;
                };

                vertices[0] = u({0, 0});
                vertices[1] = u({0, 1});
                vertices[2] = u({1, 1});
                vertices[3] = u({1, 0});

                int side_sum = 0;
                Eigen::Vector2f u_min;
                float min_distance = std::numeric_limits<float>::max();

                for (int k = 0; k < 4; ++k) {
                    Eigen::Vector2f v1 = vertices[k];
                    Eigen::Vector2f v2 = (k != 3) ? vertices[k + 1] : vertices[0];

                    float res = -v1.y() * (v2.x() - v1.x()) + v1.x() * (v2.y() - v1.y());
                    side_sum += (res >= 0 ? 1 : -1);

                    Eigen::Vector2f temp = closest_point_on_segment(v1, v2);
                    float dist = temp.norm();

                    if (dist < min_distance) {
                        min_distance = dist;
                        u_min = temp;
                    }
                }
                if (std::abs(side_sum) == 4) {
                    u_min = Eigen::Vector2f(0, 0);
                }
                float dist = std::sqrt(u_min.squaredNorm() + y.squaredNorm() - y.dot(Q * Q.transpose() * y)) - r_caps.roh - o_caps.roh;
                min_dist = std::min(min_dist, dist);
            }
        }
        return min_dist;
    }

    std::pair<std::vector<Capsule>, std::vector<Capsule>> get_capsule_pos(
        const std::vector<Eigen::Matrix4f>& fk_list,
        const std::vector<Obstacle>& obstacles) {

        std::vector<Capsule> robot_capsules;

        // Elbow (4. Frame)
        robot_capsules.push_back({
            (fk_list[4] * Eigen::Vector4f(0, 0, -0.055, 1)).head<3>(),
            (fk_list[4] * Eigen::Vector4f(0, 0, 0.055, 1)).head<3>(),
            0.075f
        });

        // Forearm 1 (5. Frame)
        robot_capsules.push_back({
            (fk_list[5] * Eigen::Vector4f(0, 0, -0.23, 1)).head<3>(),
            (fk_list[5] * Eigen::Vector4f(0, 0, -0.32, 1)).head<3>(),
            0.07f
        });

        // Forearm 2 (5. & 6. Frame)
        robot_capsules.push_back({
            (fk_list[5] * Eigen::Vector4f(0, 0.07, -0.18, 1)).head<3>(),
            (fk_list[6] * Eigen::Vector4f(0, 0, -0.1, 1)).head<3>(),
            0.045f
        });

        // Wrist (6. Frame)
        robot_capsules.push_back({
            (fk_list[6] * Eigen::Vector4f(0, 0, -0.08, 1)).head<3>(),
            (fk_list[6] * Eigen::Vector4f(0, 0, 0.01, 1)).head<3>(),
            0.067f
        });

        // Hand 1 (7. Frame)
        robot_capsules.push_back({
            (fk_list[7] * Eigen::Vector4f(0, 0, -0.04, 1)).head<3>(),
            (fk_list[7] * Eigen::Vector4f(0, 0, 0.175, 1)).head<3>(),
            0.06f
        });

        // Hand 2 (7. Frame)
        robot_capsules.push_back({
            (fk_list[7] * Eigen::Vector4f(0, 0.061, 0.13, 1)).head<3>(),
            (fk_list[7] * Eigen::Vector4f(0, -0.061, 0.13, 1)).head<3>(),
            0.06f
        });
        // Hand 3 (7. Frame)
        robot_capsules.push_back({
            (fk_list[7] * Eigen::Vector4f(0.03, 0.06, 0.085, 1)).head<3>(),
            (fk_list[7] * Eigen::Vector4f(0.06, 0.03, 0.085, 1)).head<3>(),
            0.03f
        });

        // Obstacle capsules
        std::vector<Capsule> obst_capsules;

        for (const auto& obst : obstacles) {
            Eigen::Vector3f dims_sorted(obst.size);
            std::vector<int> idx = {0, 1, 2};
            std::sort(idx.begin(), idx.end(), [&dims_sorted](int i1, int i2) { return dims_sorted[i1] < dims_sorted[i2]; });

            float l = dims_sorted[idx[2]];
            Eigen::Vector3f p(obst.pos);
            p[idx[2]] += l;
            Eigen::Vector3f u(obst.pos);
            u[idx[2]] -= l;

            obst_capsules.push_back({
                p,
                u,
                float(std::sqrt(std::pow(dims_sorted[idx[0]], 2) + std::pow(dims_sorted[idx[1]], 2)))
            });
        }
        return {robot_capsules, obst_capsules};
    }
}