#include <Eigen/Geometry>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <lidar_slam/lidar_slam.h>
#include <pcl/filters/voxel_grid.h>
#include <chrono>
#include <random>
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

    auto linearSolver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    linearSolver->setBlockOrdering(false);
    auto blockSolver = g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver));
    auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));

    optimizer_.setAlgorithm(solver);
    optimizer_.setVerbose(false);

    Eigen::Matrix3d transNoise = Eigen::Matrix3d::Zero();
    for (int i = 0; i < 3; ++i)
        transNoise(i, i) = std::pow(0.01, 2);

    Eigen::Matrix3d rotNoise = Eigen::Matrix3d::Zero();
    for (int i = 0; i < 3; ++i)
        rotNoise(i, i) = std::pow(0.001, 2);

    information_ = Eigen::Matrix<double, 6, 6>::Zero();
    information_.block<3, 3>(0, 0) = transNoise.inverse();
    information_.block<3, 3>(3, 3) = rotNoise.inverse();

    accumulated_odometry_transform_ = Eigen::Matrix4d::Identity();
}

boost::optional<Eigen::Matrix4d> LidarSlam::TryAlignment(const PointCloudPtr source,
                                                         const PointCloudPtr target,
                                                         const float max_shift_misalignment,
                                                         const float max_angle_misalignment)
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

    t1.join();
    t2.join();

    if (gicp1_.hasConverged() && gicp2_.hasConverged())
    {
        Eigen::Matrix4f result1 = gicp1_.getFinalTransformation();
        Eigen::Matrix4f result2 = gicp2_.getFinalTransformation();
        const Eigen::Matrix4f diff = result1 * result2;

        const auto misalignment = GetAbsoluteShiftAngle(diff.cast<double>());

        if (misalignment.first < max_shift_misalignment && misalignment.second < max_angle_misalignment)
        {
            // return valid value only if left-to-right misalignment is not too big
            result = result1.cast<double>();
            // Eigen::Matrix4f average = result1 * 0.5F + result2.inverse() * 0.5F;
            // result = average.cast<double>();
        }
    }

    return result;
}

std::pair<double, double> LidarSlam::GetAbsoluteShiftAngle(const Eigen::Matrix4d& matrix)
{
    const double dx = matrix(0, 3);
    const double dy = matrix(1, 3);
    const double dz = matrix(2, 3);
    const double translation = std::sqrt(dx * dx + dy * dy + dz * dz);

    const Eigen::Quaterniond angle(Eigen::Matrix3d(matrix.block(0, 0, 3, 3)));
    const double rotation =
        std::sqrt(angle.x() * angle.x() + angle.y() * angle.y() + angle.z() * angle.z()) * angle.w();

    return std::make_pair(translation, rotation);
}

void LidarSlam::LocalThreadRun()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::cout << "LidarSlam::LocalThreadRun() is running!" << std::endl;
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

            if (odometry_callback_)
            {
                std::cout << "Send initial odometry" << std::endl;
                odometry_callback_(accumulated_odometry_transform_, latest_cloud_local->header.stamp);
            }

            continue;
        }

        std::thread try_find_loop;
        int try_id{};
        g2o::VertexSE3* try_vertex;
        auto prev_vertex = *vertices_.end();
        boost::optional<Eigen::Matrix4d> try_result;

        // try to keep number of loop-closure edges not too big
        if (vertice_id_ > 10 && loopclosureEdges_.size() * 5 < odometryEdges_.size())
        {
            std::uniform_int_distribution<> rand(0, vertice_id_ - 4);
            try_id = rand(gen);

            auto try_cloud = clouds_[try_id];
            try_vertex = vertices_[try_id];

            try_find_loop = std::thread([&]() { try_result = TryAlignment(try_cloud, previous_cloud_); });
            try_find_loop.detach();
        }
        // we want to find transform from latest cloud into previous cloud
        // this shall be the same as transform from latest "frame" into previous "frame"
        std::cout << "TryAlignment ..." << std::endl;
        auto result1 = TryAlignment(previous_cloud_, latest_cloud_local);

        if (try_find_loop.joinable())
        {
            try_find_loop.join();
        }

        // "Odometry" edge has value
        if (result1.has_value())
        {
            std::cout << "SUCCESSFULL" << std::endl;
            if (vertice_id_ == 0)
            {
                auto first_vertex = new g2o::VertexSE3();
                first_vertex->setId(vertice_id_++);
                first_vertex->setToOrigin();
                first_vertex->setFixed(true);
                vertices_.push_back(first_vertex);
                clouds_.push_back(previous_cloud_);
                optimizer_.addVertex(first_vertex);
            }

            auto shiftangle = GetAbsoluteShiftAngle(result1.value());

            // if translation & rotation are not so big, we not add this latest cloud into the graph, and do not touch
            // accumulated_odometry_transform_ However, still return latest odometry
            if (shiftangle.first < 0.06 && shiftangle.second < 0.01)
            {
                if (odometry_callback_)
                {
                    odometry_callback_(result1.value() * accumulated_odometry_transform_,
                                       latest_cloud_local->header.stamp);
                }
            }
            else
            {
                accumulated_odometry_transform_ = result1.value() * accumulated_odometry_transform_;
                previous_cloud_ = latest_cloud_local;
                if (odometry_callback_)
                {
                    odometry_callback_(accumulated_odometry_transform_, latest_cloud_local->header.stamp);
                }
            }

            auto new_vertex = new g2o::VertexSE3();
            new_vertex->setId(vertice_id_++);
            new_vertex->setFixed(false);
            // new_vert->setEstimate(previous_transform_);
            vertices_.push_back(new_vertex);
            clouds_.push_back(latest_cloud_local);
            optimizer_.addVertex(new_vertex);

            auto edge = new g2o::EdgeSE3();
            edge->setId(edge_id_++);
            edge->setVertex(0, new_vertex);
            edge->setVertex(1, prev_vertex);
            Eigen::Isometry3d t(result1.value());
            edge->setMeasurement(t);
            edge->setInformation(information_);
            odometryEdges_.push_back(edge);
            optimizer_.addEdge(edge);

            if (try_result.has_value() && bool(try_vertex))
            {
                std::cout << "LOOP CLOSURE FOUND!" << std::endl;
                auto loop_edge = new g2o::EdgeSE3();
                loop_edge->setId(edge_id_++);
                loop_edge->setVertex(0, prev_vertex);
                loop_edge->setVertex(1, try_vertex);
                loop_edge->setMeasurement(Eigen::Isometry3d(try_result.value()));
                loop_edge->setInformation(information_);
                loopclosureEdges_.push_back(loop_edge);
                optimizer_.addEdge(loop_edge);
            }

            optimizer_.initializeOptimization();
            if (optimizer_.optimize(5, true) > 0)
            {
                if (mapping_callback_)
                {
                    double data[7];
                    if (new_vertex->getEstimateData(data))
                    {
                        std::cout << "Mapping Estimated Data = [" << data[0] << "," << data[1] << "," << data[2] << ",";
                        std::cout << data[3] << "," << data[4] << "," << data[5] << "," << data[6] << "]" << std::endl;

                        mapping_callback_(Eigen::Matrix4d::Identity(), latest_cloud_local->header.stamp);
                    }
                }
            }
        }
        else
        {
            std::cout << "NOT SUCCESSFULL :((" << std::endl;
        }
    }
    std::cout << "LidarSlam::LocalThreadRun() stopped!" << std::endl;
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