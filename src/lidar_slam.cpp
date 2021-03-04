#include <Eigen/Geometry>
#include <lidar_slam/lidar_slam.h>
#include <pcl/filters/voxel_grid.h>
#include <array>
#include <atomic>
#include <chrono>
#include <thread>

namespace lidar_slam
{

void LidarSlam::Run()
{
    while (processing_thread_running_)
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
            previous_transform_.row(0) << 1.0, 0.0, 0.0, 0.0;
            previous_transform_.row(1) << 0.0, 1.0, 0.0, 0.0;
            previous_transform_.row(2) << 0.0, 0.0, 1.0, 0.0;
            previous_transform_.row(3) << 0.0, 0.0, 0.0, 1.0;

            if (output_callback_)
            {
                output_callback_(previous_transform_, local_cloud->header.stamp);
            }

            continue;
        }

        // in order to be sure about correctness we check that direct and inverse alignments agree
        std::thread t1([&] {
            LidarSlam::PointCloud tmp_cloud1{};
            icp1_.setInputSource(local_cloud);
            icp1_.setInputTarget(previous_cloud_);
            icp1_.align(tmp_cloud1);
        });

        std::thread t2([&] {
            LidarSlam::PointCloud tmp_cloud2{};
            icp2_.setInputSource(previous_cloud_);
            icp2_.setInputTarget(local_cloud);
            icp2_.align(tmp_cloud2);
        });

        t1.join();
        t2.join();

        if (icp1_.hasConverged() && icp2_.hasConverged())
        {
            std::cout << "both converged" << std::endl;
            Eigen::Matrix4f result1 = icp1_.getFinalTransformation();
            Eigen::Matrix4f result2 = icp2_.getFinalTransformation();
            const Eigen::Matrix4f diff = result1 * result2.inverse();
            const float misalignment_shift = std::abs(diff(0, 3)) + std::abs(diff(1, 3)) + std::abs(diff(2, 3));
            if (misalignment_shift < 0.1F)  // we accept 10cm or less
            {
                previous_transform_ = result1 * previous_transform_;
                previous_cloud_ = local_cloud;
                if (output_callback_)
                {
                    output_callback_(previous_transform_, local_cloud->header.stamp);
                }

                std::cout << "Publishing latest transform:" << std::endl;
                std::cout << previous_transform_ << std::endl;
            }
        }
        else
        {
            std::cout << "some ICPS have not converged" << std::endl;
        }
    }
}

}  // namespace lidar_slam