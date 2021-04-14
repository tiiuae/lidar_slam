#ifndef LIDAR_SLAM_HELPERS_H
#define LIDAR_SLAM_HELPERS_H

#include <Eigen/Core>
#include <Eigen/Geometry>
namespace lidar_slam
{

class Helpers
{
  public:
    template <typename type = float>
    static Eigen::Matrix<type, 4, 4> lookAt(Eigen::Matrix<type, 3, 1> const& eye,
                                            Eigen::Matrix<type, 3, 1> const& center,
                                            Eigen::Matrix<type, 3, 1> const& up)
    {
        typedef Eigen::Matrix<type, 4, 4> Matrix4;
        typedef Eigen::Matrix<type, 3, 1> Vector3;

        Vector3 f = (center - eye).normalized();
        Vector3 u = up.normalized();
        Vector3 s = f.cross(u).normalized();
        u = s.cross(f);

        Matrix4 res;
        res << s.x(), s.y(), s.z(), -s.dot(eye), u.x(), u.y(), u.z(), -u.dot(eye), -f.x(), -f.y(), -f.z(), f.dot(eye),
            0, 0, 0, 1;

        return res;
    }

    /// @brief Returns a perspective transformation matrix like the one from gluPerspective
    /// @see http://www.opengl.org/sdk/docs/man2/xhtml/gluPerspective.xml
    /// @see glm::perspective
    template <typename Scalar = float>
    static Eigen::Matrix<Scalar, 4, 4> perspective(Scalar fovy, Scalar aspect, Scalar zNear, Scalar zFar)
    {
        Eigen::Transform<Scalar, 3, Eigen::Projective> tr;
        tr.matrix().setZero();
        assert(aspect > 0);
        assert(zFar > zNear);
        assert(zNear > 0);
        Scalar radf = M_PI * fovy / 180.0;
        Scalar tan_half_fovy = std::tan(radf / 2.0);
        tr(0, 0) = 1.0 / (aspect * tan_half_fovy);
        tr(1, 1) = 1.0 / (tan_half_fovy);
        tr(2, 2) = -(zFar + zNear) / (zFar - zNear);
        tr(3, 2) = -1.0;
        tr(2, 3) = -(2.0 * zFar * zNear) / (zFar - zNear);
        return tr.matrix();
    }

    /// @return pair of translation and rotation misalignment
    static std::pair<double, double> GetAbsoluteShiftAngle(const Eigen::Matrix4d& matrix)
    {
        const double dx = matrix(0, 3);
        const double dy = matrix(1, 3);
        const double dz = matrix(2, 3);
        const double translation = std::sqrt(dx * dx + dy * dy + dz * dz);

        const Eigen::Quaterniond angle(Eigen::Matrix3d(matrix.block(0, 0, 3, 3)));
        const double rotation =
            std::sqrt(angle.x() * angle.x() + angle.y() * angle.y() + angle.z() * angle.z()) * angle.w();

        return std::make_pair(translation, rotation * 2.);
    }
};

} //namespace lidar_slam

#endif  // LIDAR_SLAM_HELPERS_H
