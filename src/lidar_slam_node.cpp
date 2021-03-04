#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/gicp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <thread>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>
#include <Eigen/Geometry>

//#include <tf2/convert.h>
//#include <tf2_ros/buffer.h>
//#include <tf2_ros/transform_listener.h>

using TransformStamped = geometry_msgs::msg::TransformStamped;

class LidarSlamNode : public rclcpp::Node
{
  public:
    using PointCloudMsg = sensor_msgs::msg::PointCloud2;
    using Point = pcl::PointXYZ;
    using PointCloudPCL = pcl::PointCloud<Point>;
    using PointCloudPtr = PointCloudPCL::Ptr;

    //static const std::size_t BufferSize = 5U;

    LidarSlamNode()
        : Node("lidar_slam"),
          cloud_subscriber_{},
          previous_cloud_pcl_{},
          // buffer_{},
          // buffer_head_{},
          // buffer_tail_{},
          // buffer_mutex_{},
          icp1_(), icp2_(),
          latest_cloud_msg_{},
          processing_thread_{},
          processing_thread_running_{}
    {
        declare_parameter<std::string>("cloud_topic", "/camera/depth/color/points");
        declare_parameter<std::string>("base_frame_id", "base_link");
        declare_parameter<std::string>("map_frame_id", "map");

        const std::string cloud_topic = get_parameter("cloud_topic").as_string();
        base_frame_ = get_parameter("base_frame_id").as_string();
        map_frame_ = get_parameter("map_frame_id").as_string();

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        processing_thread_running_ = true;
        processing_thread_ = std::thread(&LidarSlamNode::Run, this);

        rclcpp::QoS qos(5);
        qos = qos.best_effort();
        cloud_subscriber_ = create_subscription<PointCloudMsg>(
            cloud_topic, qos, std::bind(&LidarSlamNode::GrabPointCloud, this, std::placeholders::_1));

        RCLCPP_INFO(get_logger(), "Subscribed to topic '" + cloud_topic + "'");
    }

    ~LidarSlamNode()
    {
        processing_thread_running_ = false;
        processing_thread_.join();
    }

  private:
    rclcpp::Subscription<PointCloudMsg>::SharedPtr cloud_subscriber_;
    std::string base_frame_, map_frame_;
    pcl::GeneralizedIterativeClosestPoint<Point, Point> icp1_;
    pcl::GeneralizedIterativeClosestPoint<Point, Point> icp2_;

    PointCloudPtr previous_cloud_pcl_;
    PointCloudMsg::SharedPtr latest_cloud_msg_;

    Eigen::Matrix4f previous_transform_;

    // std::array<CloudMsg::SharedPtr, BufferSize> buffer_;
    // std::size_t buffer_head_,buffer_tail_;
    std::mutex buffer_mutex_;
    std::thread processing_thread_;
    std::atomic_bool processing_thread_running_;

    // tf2_ros::Buffer tf_buffer_;
    // tf2_ros::TransformListener tf_listener_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    void GrabPointCloud(const PointCloudMsg::SharedPtr msg)
    {
        RCLCPP_INFO(get_logger(), "Recieved cloud at time " + std::to_string(msg->header.stamp.sec));
        const std::lock_guard<std::mutex> guard(buffer_mutex_);
        latest_cloud_msg_ = msg;
    }

    void Run()
    {
        while (processing_thread_running_)
        {
            PointCloudMsg::SharedPtr local_msg;
            {
                const std::lock_guard<std::mutex> guard(buffer_mutex_);
                local_msg = latest_cloud_msg_;
                latest_cloud_msg_.reset();
            }

            if (!local_msg)
            {
                std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(1));
                continue;
            }

            // on the first cycle previous_cloud_pcl_ is empty
            if (!previous_cloud_pcl_)
            {
                previous_cloud_pcl_.reset(new PointCloudPCL());
                pcl::moveFromROSMsg(*local_msg, *previous_cloud_pcl_);

                previous_transform_.row(0) << 1.0, 0.0, 0.0, 0.0;
                previous_transform_.row(1) << 0.0, 1.0, 0.0, 0.0;
                previous_transform_.row(2) << 0.0, 0.0, 1.0, 0.0;
                previous_transform_.row(3) << 0.0, 0.0, 0.0, 1.0;

                PublishTransform(previous_transform_, local_msg->header.stamp);
                std::cout << "Publishing first transform:" << std::endl;
                std::cout << previous_transform_ << std::endl;
                continue;
            }
            else
            {
                const PointCloudPtr latest_pcl_cloud(new PointCloudPCL());
                pcl::moveFromROSMsg(*local_msg, *latest_pcl_cloud);

                // in order to be sure about correctness we check that direct and inverse alignments agree
#pragma omp parallel sections
                {
#pragma omp section
                    {
                        PointCloudPCL tmp_cloud1;
                        icp1_.setInputSource(latest_pcl_cloud);
                        icp1_.setInputTarget(previous_cloud_pcl_);
                        icp1_.align(tmp_cloud1);
                    }
#pragma omp section
                    {
                        PointCloudPCL tmp_cloud2;
                        icp2_.setInputSource(previous_cloud_pcl_);
                        icp2_.setInputTarget(latest_pcl_cloud);
                        icp2_.align(tmp_cloud2);
                    }
                };

                if(icp1_.hasConverged() && icp2_.hasConverged())
                {
                    std::cout << "both converged" << std::endl;
                    Eigen::Matrix4f result1 = icp1_.getFinalTransformation();
                    Eigen::Matrix4f result2 = icp2_.getFinalTransformation();
                    const Eigen::Matrix4f diff = result1 * result2.inverse();
                    const float shift = std::abs(diff(0, 3)) + std::abs(diff(1, 3)) + std::abs(diff(2, 3));
                    if(shift < 0.1F) // 10cm
                    {
                        previous_transform_ = result1 * previous_transform_;
                        previous_cloud_pcl_ = latest_pcl_cloud;
                        PublishTransform(previous_transform_, local_msg->header.stamp);
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
    }

    void PublishTransform(const Eigen::Matrix4f& transform, const builtin_interfaces::msg::Time stamp)
    {
        Eigen::Matrix3f rot = transform.block(0, 0, 3, 3);
        Eigen::Quaternionf q(rot);

        TransformStamped msg;
        msg.transform.translation.x = transform(0, 3);
        msg.transform.translation.y = transform(1, 3);
        msg.transform.translation.z = transform(2, 3);

        msg.transform.rotation.x = q.x();
        msg.transform.rotation.y = q.y();
        msg.transform.rotation.z = q.z();
        msg.transform.rotation.w = q.w();

        msg.header.stamp = stamp;
        msg.header.frame_id = map_frame_;
        msg.child_frame_id = base_frame_;
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<LidarSlamNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}
