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

LidarSlam::LidarSlam(const LidarSlamParameters& params)
    : params_{params}
{
    Eigen::Matrix3d transNoise = Eigen::Matrix3d::Zero();
    for (int i = 0; i < 3; ++i)
        transNoise(i, i) = std::pow(params.gicp_translation_noise, 2);

    Eigen::Matrix3d rotNoise = Eigen::Matrix3d::Zero();
    for (int i = 0; i < 3; ++i)
        rotNoise(i, i) = std::pow(params.gicp_rotation_noise, 2);

    icp_information_ = Eigen::Matrix<double, 6, 6>::Zero();
    icp_information_.block<3, 3>(0, 0) = transNoise.inverse();
    icp_information_.block<3, 3>(3, 3) = rotNoise.inverse();

    if (params_.automatic_start)
    {
        Start();
    }
}

void LidarSlam::LoopClousuresThread()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    fast_gicp::FastVGICP<Point, Point> loopclosure_gicp_;
    loopclosure_gicp_.setResolution(params_.gicp_resolution);
    loopclosure_gicp_.setVoxelAccumulationMode(fast_gicp::VoxelAccumulationMode::ADDITIVE_WEIGHTED);
    loopclosure_gicp_.setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT1);
    loopclosure_gicp_.setRegularizationMethod(fast_gicp::RegularizationMethod::NONE);

//    SLAM_LOG << "LidarSlam::LoopClousuresThread() is running!" << std::endl;
//    while (loopclousures_thread_running_)
//    {
//        std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(10));
//    }
//    SLAM_LOG << "LidarSlam::LoopClousuresThread() stopped!" << std::endl;
}

void LidarSlam::OdometryThread()
{
    LidarSlam::PointCloud tmp_cloud1{};
    Eigen::Matrix4d accumulated_odometry_ = Eigen::Matrix4d::Identity();
    PointCloudPtr previous_cloud_;

    fast_gicp::FastVGICP<Point, Point> odometry_gicp_;
    odometry_gicp_.setResolution(params_.gicp_resolution);
    odometry_gicp_.setVoxelAccumulationMode(fast_gicp::VoxelAccumulationMode::ADDITIVE_WEIGHTED);
    odometry_gicp_.setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT1);
    odometry_gicp_.setRegularizationMethod(fast_gicp::RegularizationMethod::NONE);

    // Adding vertex to the edge (edge->setVertex(..)) changes vertex too, which is not thread-safe
    // So we cannot add last_odometry_vertex_ directly to the Graph, but rather wait for the next odometry step
    // During this time we keep "current" odometry vertex here
    CloudVertexSE3* last_odometry_vertex {nullptr};

    SLAM_LOG << "LidarSlam::OdometryThread() is running!" << std::endl;
    while (odometry_thread_running_)
    {
        // swap whatever latest_cloud_ we have into local variable
        PointCloud::Ptr latest_cloud_local;
        {
            const std::lock_guard<std::mutex> guard(latest_cloud_mutex_);
            latest_cloud_local.swap(latest_cloud_);
        }

        if (!latest_cloud_local || (latest_cloud_local->size() < 20U))
        {
            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(5));
            continue;
        }

        // initialize previous_cloud_ as well as first vertex whenever non-empty cloud arrives
        if (!previous_cloud_)
        {
            previous_cloud_ = latest_cloud_local;
            odometry_gicp_.setInputTarget(previous_cloud_);
            last_odometry_vertex = new CloudVertexSE3(previous_cloud_);
            last_odometry_vertex->setId(vertice_id_++);
            last_odometry_vertex->setToOrigin();
            last_odometry_vertex->setFixed(true);
            last_odometry_vertex->setMarginalized(true);
            //clouds_[last_odometry_vertex->id()] = previous_cloud_;

            if (odometry_callback_)
            {
                odometry_callback_(accumulated_odometry_, latest_cloud_local->header.stamp);
            }
            continue;
        }

        odometry_gicp_.setInputSource(latest_cloud_local);
        odometry_gicp_.align(tmp_cloud1);
        if (odometry_gicp_.hasConverged())
        {
            const Eigen::Matrix4d result = odometry_gicp_.getFinalTransformation().cast<double>();

            SLAM_LOG << "ICP converged to:" << std::endl;
            SLAM_LOG << result << std::endl;

            auto shiftangle = GetAbsoluteShiftAngle(result);
            if (shiftangle.first < 0.1 && shiftangle.second < 0.01)
            {
                // translation & rotation are small, so we don't add this cloud into the graph,
                // neither change previous_cloud_
                // However, still return latest odometry
                if (odometry_callback_)
                {
                    odometry_callback_(accumulated_odometry_ * result, latest_cloud_local->header.stamp);
                }
            }
            else
            {
                // motion is significant
                odometry_gicp_.swapSourceAndTarget();  // keep pre-calculated voxels/kd-tree
                accumulated_odometry_ = accumulated_odometry_ * result;
                previous_cloud_ = latest_cloud_local;

                if (odometry_callback_)
                {
                    odometry_callback_(accumulated_odometry_, latest_cloud_local->header.stamp);
                }

                Eigen::Isometry3d odom(accumulated_odometry_);

                // we also add Vertex into the Graph
                auto vertex = new CloudVertexSE3(previous_cloud_);
                vertex->setId(vertice_id_++);
                vertex->setFixed(false);
                vertex->setMarginalized(true);
                vertex->setEstimate(odom);

                auto edge = new g2o::EdgeSE3();
                edge->setId(edge_id_++);
                edge->setVertex(0, last_odometry_vertex);
                edge->setVertex(1, vertex);
                Eigen::Isometry3d t(result);
                edge->setMeasurement(t);
                edge->setInformation(icp_information_);

                //clouds_[vertex->id()] = previous_cloud_;
                {
                    const std::lock_guard<std::mutex> guard(vertices_mutex_);
                    vertices_.push_back(last_odometry_vertex);
                    odometryEdges_.push_back(edge);
                }

                last_odometry_vertex = vertex;
            }
        }
        else
        {
            SLAM_LOG << "ICP haven't converged" << std::endl;
        }

        //        std::thread try_find_loop;
        //        int try_id{};
        //        g2o::VertexSE3* try_vertex = nullptr;
        //        g2o::VertexSE3* prev_vertex = nullptr;
        //        boost::optional<Eigen::Matrix4d> try_result;
        //
        //        // try to keep number of loop-closure edges not too big
        //        if (vertice_id_ > 10 && loopclosureEdges_.size() * 5 < odometryEdges_.size())
        //        {
        //            std::uniform_int_distribution<> rand(0, vertice_id_ - 4);
        //            try_id = rand(gen);
        //
        //            prev_vertex = *vertices_.end();
        //            auto try_cloud = clouds_[try_id];
        //            try_vertex = vertices_[try_id];
        //
        //            try_find_loop = std::thread([&]() { try_result = TryAlignment(try_cloud, previous_cloud_); });
        //            try_find_loop.detach();
        //        }
        // we want to find transform from latest cloud into previous cloud
        // this shall be the same as transform from latest "frame" into previous "frame"
        //        std::cout << "TryAlignment ..." << std::endl;
        //        auto result1 = TryAlignment(latest_cloud_local, previous_cloud_);

        //        if (try_find_loop.joinable())
        //        {
        //            try_find_loop.join();
        //        }

        // "Odometry" edge has value
        //        if (result.has_value())
        //        {
        //            std::cout << "SUCCESSFULL" << std::endl;
        //
        //
        //
        //            const Eigen::Matrix4d previous_odometry = accumulated_odometry_;

        // if translation & rotation are not so big, we not add this latest cloud into the graph, and do not touch
        // accumulated_odometry_ However, still return latest odometry
        //            if (shiftangle.first < 0.06 && shiftangle.second < 0.01)
        //            {

        //                auto new_vertex = new g2o::VertexSE3();
        //                new_vertex->setId(vertice_id_++);
        //                new_vertex->setFixed(false);
        //                // new_vert->setEstimate(previous_transform_);
        //                vertices_.push_back(new_vertex);
        //                clouds_.push_back(latest_cloud_local);
        //                optimizer_.addVertex(new_vertex);
        //
        //                auto edge = new g2o::EdgeSE3();
        //                edge->setId(edge_id_++);
        //                edge->setVertex(0, new_vertex);
        //                edge->setVertex(1, prev_vertex);
        //                Eigen::Isometry3d t(result1.value());
        //                edge->setMeasurement(t);
        //                edge->setInformation(information_);
        //                odometryEdges_.push_back(edge);
        //                optimizer_.addEdge(edge);

        //                if (odometry_callback_)
        //                {
        //                    odometry_callback_(result.value() * accumulated_odometry_,
        //                                       latest_cloud_local->header.stamp);
        //                }
        //            }
        //            else
        //            {
        //                accumulated_odometry_ = result.value() * accumulated_odometry_;
        //                previous_cloud_ = latest_cloud_local;
        //                if (odometry_callback_)
        //                {
        //                    odometry_callback_(accumulated_odometry_, latest_cloud_local->header.stamp);
        //                }
        //            }

        //
        //            if (try_result.has_value() && (try_vertex != nullptr))
        //            {
        //                std::cout << "LOOP CLOSURE FOUND!" << std::endl;
        //                auto loop_edge = new g2o::EdgeSE3();
        //                loop_edge->setId(edge_id_++);
        //                loop_edge->setVertex(0, prev_vertex);
        //                loop_edge->setVertex(1, try_vertex);
        //                loop_edge->setMeasurement(Eigen::Isometry3d(try_result.value()));
        //                loop_edge->setInformation(information_);
        //                loopclosureEdges_.push_back(loop_edge);
        //                optimizer_.addEdge(loop_edge);
        //            }

        //        }
        //        else
        //        {
        //            std::cout << "NOT SUCCESSFULL :((" << std::endl;
        //        }
    }
    SLAM_LOG << "LidarSlam::OdometryThread() stopped!" << std::endl;
}

void LidarSlam::MappingThread()
{

    auto linearSolver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    linearSolver->setBlockOrdering(false);
    auto blockSolver = g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver));
    auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));

    /// Graph Optimizer
    g2o::SparseOptimizer optimizer_;
    optimizer_.setAlgorithm(solver);
    optimizer_.setVerbose(params_.optimizer_verbose);
    SLAM_LOG << "LidarSlam::MappingThread() is running!" << std::endl;


    CloudVertexSE3* prev_vertex = nullptr;

    while (mapping_thread_running_)
    {
        std::vector<CloudVertexSE3*> vertices{};
        std::vector<g2o::EdgeSE3*> odometryEdges{};
        std::vector<g2o::EdgeSE3*> loopclosureEdges{};

        {
            const std::lock_guard<std::mutex> guard(vertices_mutex_);
            vertices.swap(vertices_);
            odometryEdges.swap(odometryEdges_);
            loopclosureEdges.swap(loopclosureEdges_);
        }

        if (!vertices.empty())
        {
            for (auto vertex : vertices)
            {
                vertex->setMarginalized(false);
                optimizer_.addVertex(vertex);
                prev_vertex = vertex;
            }
        }

        if (!odometryEdges.empty())
        {
            for (auto edge : odometryEdges)
            {
                optimizer_.addEdge(edge);
            }
        }

        if (optimizer_.initializeOptimization())
        {
            if (optimizer_.optimize(5, true) > 0)
            {
                if (mapping_callback_)
                {
                    const Eigen::Isometry3d p = prev_vertex->estimate();
//                    const Eigen::Matrix4d odometry2mapping = p.matrix() * previous_odometry;
//
//                    const auto t = p.translation();
//                    const auto r = p.rotation();
//
//                    std::cout << "Mapping: " << std::endl;
//                    std::cout << p.matrix() << std::endl;
//
//                    std::cout << "Odometry: " << std::endl;
//                    std::cout << previous_odometry << std::endl;
//
//                    std::cout << "Odometry-to-Mapping: " << std::endl;
//                    std::cout << odometry2mapping << std::endl;

                    mapping_callback_(p.matrix(), prev_vertex->cloud->header.stamp);
                }
            }
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(50));
        }
    }

    SLAM_LOG << "LidarSlam::MappingThread() stopped!" << std::endl;
}

void LidarSlam::Start()
{
    if (!odometry_thread_running_ && !odometry_thread_.joinable())
    {
        odometry_thread_running_ = true;
        odometry_thread_ = std::thread(&LidarSlam::OdometryThread, this);
    }
    if (loopclousures_thread_running_ && !loopclousures_thread_.joinable())
    {
        loopclousures_thread_running_ = true;
        loopclousures_thread_ = std::thread(&LidarSlam::LoopClousuresThread, this);
    }
    if (mapping_thread_running_ && !mapping_thread_.joinable())
    {
        mapping_thread_running_ = true;
        mapping_thread_ = std::thread(&LidarSlam::MappingThread, this);
    }
}

void LidarSlam::Stop()
{
    odometry_thread_running_ = false;
    mapping_thread_running_ = false;
    loopclousures_thread_running_ = false;
    if (odometry_thread_.joinable())
    {
        odometry_thread_.join();
    }
    if (mapping_thread_.joinable())
    {
        mapping_thread_.join();
    }
    if (loopclousures_thread_.joinable())
    {
        loopclousures_thread_.join();
    }
}

LidarSlam::~LidarSlam()
{
    Stop();
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

}  // namespace lidar_slam