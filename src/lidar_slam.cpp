#include <Eigen/Geometry>
#include <lidar_slam/lidar_slam.h>
#include <pcl/filters/voxel_grid.h>
#include <chrono>
#include <thread>
// following includes come last
#include <fast_gicp/gicp/impl/fast_gicp_impl.hpp>
#include <fast_gicp/gicp/impl/fast_vgicp_impl.hpp>
#include <fast_gicp/gicp/impl/lsq_registration_impl.hpp>

template class fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ>;
template class fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>;
template class fast_gicp::LsqRegistration<pcl::PointXYZ, pcl::PointXYZ>;

namespace lidar_slam
{

LidarSlam::LidarSlam()
    : odometry_callback_{},
      gicp1_(),
      gicp2_(),
      previous_cloud_{},
      latest_cloud_{},
      local_thread_{},
      local_thread_running_{}
{
    gicp1_.setResolution(0.2);
    gicp1_.setVoxelAccumulationMode(fast_gicp::VoxelAccumulationMode::ADDITIVE);
    gicp1_.setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
    gicp1_.setRegularizationMethod(fast_gicp::RegularizationMethod::NONE);

    gicp2_.setResolution(0.2);
    gicp2_.setVoxelAccumulationMode(fast_gicp::VoxelAccumulationMode::ADDITIVE);
    gicp2_.setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
    gicp2_.setRegularizationMethod(fast_gicp::RegularizationMethod::NONE);

    local_thread_running_ = true;
    local_thread_ = std::thread(&LidarSlam::LocalThreadRun, this);
}

void LidarSlam::LocalThreadRun()
{
    while (local_thread_running_)
    {
        // swap whatever latest_cloud_ we have into local variable
        PointCloud::Ptr local_cloud;
        {
            const std::lock_guard<std::mutex> guard(buffer_mutex_);
            local_cloud.swap(latest_cloud_);
        }

        if (!local_cloud || local_cloud->empty())
        {
            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(1));
            continue;
        }

        // on the first cycle previous_cloud_ is empty
        if (!previous_cloud_ || previous_cloud_->empty())
        {
            previous_cloud_ = local_cloud;
            accumulated_odometry_transform_.row(0) << 1.0, 0.0, 0.0, 0.0;
            accumulated_odometry_transform_.row(1) << 0.0, 1.0, 0.0, 0.0;
            accumulated_odometry_transform_.row(2) << 0.0, 0.0, 1.0, 0.0;
            accumulated_odometry_transform_.row(3) << 0.0, 0.0, 0.0, 1.0;

            if (odometry_callback_)
            {
                odometry_callback_(accumulated_odometry_transform_, local_cloud->header.stamp);
            }

            continue;
        }

        std::thread t1([&] {
            {
                LidarSlam::PointCloud tmp{};
                gicp1_.setInputSource(previous_cloud_);
                gicp1_.setInputTarget(local_cloud);
                gicp1_.align(tmp);
            }
        });

        std::thread t2([&] {
            {
                LidarSlam::PointCloud tmp{};
                gicp2_.setInputSource(previous_cloud_);
                gicp2_.setInputTarget(local_cloud);
                gicp2_.align(tmp);
            }
        });

        t1.detach();
        t2.detach();
        t1.join();
        t2.join();

        if (gicp1_.hasConverged() && gicp2_.hasConverged())
        {
            const Eigen::Matrix4f previous_to_current = gicp1_.getFinalTransformation();
            const Eigen::Matrix4f current_to_previous = gicp2_.getFinalTransformation();
            const Eigen::Matrix4f diff = previous_to_current * current_to_previous.inverse();

            const float dx = diff(0, 3);
            const float dy = diff(1, 3);
            const float dz = diff(2, 3);
            const float translation_misalignment = std::sqrt(dx * dx + dy * dy + dz * dz);

            const Eigen::Quaternionf rotation_misalignment(Eigen::Matrix3f(diff.block(0, 0, 3, 3)));

            std::cout << "Estmated previous_to_current transform:" << std::endl;
            std::cout << previous_to_current << std::endl;
            std::cout << "Estmated current_to_previous transform:" << std::endl;
            std::cout << current_to_previous << std::endl;
            std::cout << "translation_misalignment = " << translation_misalignment << std::endl;
            std::cout << "rotation_misalignment = " << rotation_misalignment.w() << std::endl;

            if (translation_misalignment < 0.05F)
            {
                // ICP returns transform which maps points from the previous frame into current one
                // Thus, accumulated_odometry contains "drifty" transform from the very-first frame into last (current)
                // one
                accumulated_odometry_transform_ = accumulated_odometry_transform_ * previous_to_current;

                std::cout << "Accumulated transform:" << std::endl;
                std::cout << accumulated_odometry_transform_ << std::endl;

                if (odometry_callback_)
                {
                    odometry_callback_(accumulated_odometry_transform_, local_cloud->header.stamp);
                }
            }
        }
    }
}

#ifdef USE_OWN_FEATURES
void LidarSlam::FeatureDetector(const PointCloud& cloud)
{
    if (cloud.width == 1 || cloud.height == 1)
    {
        return;
    }

    struct PlanarFeature
    {
        float x, y, z;        // centroid coordinates
        float a, b, c;        // estimated "plane" coordinates: z = ax + by + c
        Eigen::Vector3f ZXt;  // Z X^T
        Eigen::Matrix3f XXt;  //
        std::size_t amount;
    };

    std::size_t width = cloud.width;
    std::size_t height = cloud.height;
    std::vector<std::vector<PlanarFeature>> features{};

    for (std::size_t s = 0; s < FeatureScales.size(); s++)
    {
        const float scale = FeatureScales[s];
        const std::size_t new_width = (width - 1U) / 2U;
        const std::size_t new_height = (height - 1U) / 2U;
        std::vector<PlanarFeature> planes0(new_width * new_height);
        features.push_back(planes0);

        // left-to-right pass
        for (std::size_t new_y = 0U; new_y < new_height; new_y++)
        {
            for (std::size_t ox = 0U; ox < width - 1U; ox++)
            {
            }
        }
    }
}
#endif
}  // namespace lidar_slam