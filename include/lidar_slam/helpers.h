#ifndef LIDAR_SLAM_HELPERS_H
#define LIDAR_SLAM_HELPERS_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/extract_indices.h>

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

    template<class Point>
    static boost::optional<Eigen::Matrix4f> Detect3DCorner(boost::shared_ptr<pcl::PointCloud<Point>> cloud, const float threshold = 0.1)
    {
        using PointCloud = pcl::PointCloud<Point>;
        using PointCloudPtr = boost::shared_ptr<pcl::PointCloud<Point>>;
        using PlaneModel = pcl::SampleConsensusModelPlane<Point>;
        using Ransac = pcl::RandomSampleConsensus<Point>;
        boost::optional<Eigen::Matrix4f> output;

        PointCloudPtr removed1(new PointCloud());
        PointCloudPtr removed2(new PointCloud());
        boost::shared_ptr<std::vector<int>> inliers(new std::vector<int>());

        if(cloud->size()  < 40)
        {
            return output;
        }


        pcl::ExtractIndices<Point> extract;
        typename PlaneModel::Ptr planeModel(new PlaneModel(cloud));
        typename Ransac::Ptr ransac(new Ransac(planeModel, threshold));
        Eigen::VectorXf plane1, plane2, plane3, ground;
        ransac->setMaxIterations(100);

        // Find first plane
        if(!ransac->computeModel())
        {
            return output;
        }

        planeModel->optimizeModelCoefficients(ransac->inliers_, ransac->model_coefficients_, plane1);
        inliers->swap(ransac->inliers_);

        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);  // Remove points, belonging to recently found plane
        extract.filter(*removed1);

        if(removed1->size() < 30)
        {
            return output;
        }

        // Find second plane
        planeModel.reset(new PlaneModel(removed1));
        ransac.reset(new Ransac(planeModel, threshold));
        if(!ransac->computeModel())
        {
            return output;
        }
        planeModel->optimizeModelCoefficients(ransac->inliers_, ransac->model_coefficients_, plane2);
        inliers->swap(ransac->inliers_);
        extract.setInputCloud(removed1);
        extract.setIndices(inliers);
        extract.setNegative(true);  // Remove points again
        extract.filter(*removed2);

        if(removed2->size() < 20)
        {
            return output;
        }

        // Find last plane
        planeModel.reset(new PlaneModel(removed2));
        ransac.reset(new Ransac(planeModel, threshold));
        if(!ransac->computeModel())
        {
            return output;
        }
        planeModel->optimizeModelCoefficients(ransac->inliers_, ransac->model_coefficients_, plane3);
        if(ransac->inliers_.size() < 10)
        {
            return output;
        }

        Eigen::Vector4f up(0, 0, 1, 0);
        const float d1 = plane1.transpose() * up;
        const float d2 = plane2.transpose() * up;
        const float d3 = plane3.transpose() * up;

        if (fabs(d3) > fabs(d2) && fabs(d3) > fabs(d1))
        {
            ground = plane3;
        }
        else if (fabs(d1) > fabs(d2) && fabs(d1) > fabs(d3))
        {
            ground = plane1;
            plane1 = plane3;
        }
        else
        {
            ground = plane2;
            plane2 = plane3;
        }

        // set plane equation such that d (plane[3]) is positive
        plane1 = plane1[3] > 0 ? plane1 : -plane1;
        plane2 = plane2[3] > 0 ? plane2 : -plane2;

        // normals of all planes
        Eigen::Vector3f n1(plane1[0], plane1[1], plane1[2]);
        Eigen::Vector3f n2(plane2[0], plane2[1], plane2[2]);
        Eigen::Vector3f ng(ground[0], ground[1], ground[2]);

        Eigen::Matrix4f result = Eigen::Matrix4f::Identity();

        // first translation is always positive
        // second translation is always negative
        // which gives us only 2 options actually.
        // we choose the one, giving valid right-handed coordinate system
        if (ng.transpose() * ((n1).cross(-n2)) > 0)
        {
            result.row(0) = plane1;
            result.row(1) = -plane2;
            result.row(2) = ground;
        }
        else
        {
            result.row(0) = plane2;
            result.row(1) = -plane1;
            result.row(2) = ground;
        }

        // Orthonormalize rotation
        result.block(0, 0, 3, 3) = Eigen::AngleAxisf(Eigen::Matrix3f(result.block(0, 0, 3, 3))).toRotationMatrix();
        output = result.inverse();
        return output;
    }


};

} //namespace lidar_slam

#endif  // LIDAR_SLAM_HELPERS_H
