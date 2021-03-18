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
      mapping_callback_{},
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

    optimizer_.setVerbose(true);

    linearSolver_ = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();

    solver_ = std::make_unique<g2o::OptimizationAlgorithmLevenberg>(
        g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver_)));

    optimizer_.setAlgorithm(solver_.get());

    Eigen::Matrix3d transNoise = Eigen::Matrix3d::Zero();
    for (int i = 0; i < 3; ++i)
        transNoise(i, i) = std::pow(0.01, 2);

    Eigen::Matrix3d rotNoise = Eigen::Matrix3d::Zero();
    for (int i = 0; i < 3; ++i)
        rotNoise(i, i) = std::pow(0.001, 2);

    information_ = Eigen::Matrix<double, 6, 6>::Zero();
    information_.block<3, 3>(0, 0) = transNoise.inverse();
    information_.block<3, 3>(3, 3) = rotNoise.inverse();
}

boost::optional<Eigen::Matrix4d> LidarSlam::TryAlignment(const PointCloudPtr source,
                                                         const PointCloudPtr target,
                                                         const float max_misalignment)
{
    boost::optional<Eigen::Matrix4d> result;

    // in order to be sure about correct convergence we check if direct and inverse alignments agree
    std::thread t1([&] {
        LidarSlam::PointCloud tmp_cloud1{};
        gicp1_.setInputSource(source);
        gicp1_.setInputTarget(target);
        gicp1_.align(tmp_cloud1);
    });

    std::thread t2([&] {
        LidarSlam::PointCloud tmp_cloud2{};
        gicp2_.setInputSource(target);
        gicp2_.setInputTarget(source);
        gicp2_.align(tmp_cloud2);
    });

    t1.detach();
    t2.detach();

    t1.join();
    t2.join();

    if (gicp1_.hasConverged() && gicp2_.hasConverged())
    {
        Eigen::Matrix4f result1 = gicp1_.getFinalTransformation();
        Eigen::Matrix4f result2 = gicp2_.getFinalTransformation();
        const Eigen::Matrix4f diff = result1 * result2.inverse();
        const float dx = diff(0, 3);
        const float dy = diff(1, 3);
        const float dz = diff(2, 3);
        const float misalignment_shift = std::sqrt(dx * dx + dy * dy + dz * dz);
        // const Eigen::Quaternionf misalignment_angle(Eigen::Matrix3f(diff.block(0, 0, 3, 3)));

        if (misalignment_shift < max_misalignment)
        {
            // return valid value only if left-to-right misalignment is not too big
            result = result1.cast<double>();
        }
    }

    return result;
}

void LidarSlam::LocalThreadRun()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    while (local_thread_running_)
    {
        // swap whatever latest_cloud_ we have into local variable
        PointCloud::Ptr latest_cloud_local;
        {
            const std::lock_guard<std::mutex> guard(buffer_mutex_);
            latest_cloud_local.swap(latest_cloud_);
        }

        if (!latest_cloud_local || latest_cloud_local->empty())
        {
            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(1));
            continue;
        }

        // on the first cycle previous_cloud_ is empty
        if (!previous_cloud_ || previous_cloud_->empty())
        {
            previous_cloud_ = latest_cloud_local;
            accumulated_odometry_transform_.row(0) << 1.0, 0.0, 0.0, 0.0;
            accumulated_odometry_transform_.row(1) << 0.0, 1.0, 0.0, 0.0;
            accumulated_odometry_transform_.row(2) << 0.0, 0.0, 1.0, 0.0;
            accumulated_odometry_transform_.row(3) << 0.0, 0.0, 0.0, 1.0;

            if (odometry_callback_)
            {
                odometry_callback_(accumulated_odometry_transform_, latest_cloud_local->header.stamp);
            }

            continue;
        }

        std::thread try_find_loop;
        int try_id{};
        std::shared_ptr<g2o::VertexSE3> try_vertex;
        boost::optional<Eigen::Matrix4d> try_result;

        // try to keep number of loop-closure edges not too big
        if (vertice_id_ > 10 && loopclosureEdges_.size() * 5 < odometryEdges_.size())
        {
            std::uniform_int_distribution<> rand(0, vertice_id_ - 4);
            try_id = rand(gen);

            auto try_cloud = clouds_[try_id];
            try_vertex = vertices_[try_id];
            // prev_vertex = vertices_[vertice_id_];

            try_find_loop = std::thread([&]() { try_result = TryAlignment(try_cloud, previous_cloud_); });
            try_find_loop.detach();
        }
        // we want to find transform from latest cloud into previous cloud
        // this shall be the same as transform from latest "frame" into previous "frame"
        auto result1 = TryAlignment(previous_cloud_, latest_cloud_local);

        if (try_find_loop.joinable())
        {
            try_find_loop.join();
        }

        // "Odometry" edge has value
        if (result1.has_value())
        {
            if (vertice_id_ == 0)
            {
                auto vert0 = std::make_shared<g2o::VertexSE3>();
                vert0->setId(vertice_id_++);
                vert0->setToOrigin();
                vert0->setFixed(true);
                vertices_.push_back(vert0);
                clouds_.push_back(previous_cloud_);
                g2o::HyperGraph::VertexSet vertexSet{vert0.get()};
                optimizer_.push(vertexSet);
            }

            accumulated_odometry_transform_ = result1.value() * accumulated_odometry_transform_;
            previous_cloud_ = latest_cloud_local;
            if (odometry_callback_)
            {
                odometry_callback_(accumulated_odometry_transform_, latest_cloud_local->header.stamp);
            }

            auto prev_vert = *vertices_.end();
            auto new_vert = std::make_shared<g2o::VertexSE3>();
            new_vert->setId(vertice_id_++);
            // new_vert->setEstimate(previous_transform_);
            vertices_.push_back(new_vert);
            clouds_.push_back(latest_cloud_local);
            g2o::HyperGraph::VertexSet vertexSet{new_vert.get()};
            optimizer_.push(vertexSet);

            auto edge = std::make_shared<g2o::EdgeSE3>();
            edge->setId(edge_id_++);
            edge->setVertex(0, new_vert.get());
            edge->setVertex(1, prev_vert.get());
            Eigen::Isometry3d t(result1.value());
            edge->setMeasurement(t);
            edge->setInformation(information_);
            odometryEdges_.push_back(edge);

            if (try_result.has_value() && bool(try_vertex))
            {
                auto loop_edge = std::make_shared<g2o::EdgeSE3>();
                loop_edge->setId(edge_id_++);
                loop_edge->setVertex(0, prev_vert.get());
                loop_edge->setVertex(1, try_vertex.get());
                loop_edge->setMeasurement(Eigen::Isometry3d(try_result.value()));
                loop_edge->setInformation(information_);
                loopclosureEdges_.push_back(loop_edge);
            }

            optimizer_.initializeOptimization();
            if (optimizer_.optimize(2, true) > 0)
            {
                if (mapping_callback_)
                {
                    double data[7];
                    if (new_vert->getEstimateData(data))
                    {

                        mapping_callback_(accumulated_odometry_transform_, latest_cloud_local->header.stamp);
                    }
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