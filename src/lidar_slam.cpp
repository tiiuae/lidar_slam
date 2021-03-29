#include <Eigen/Geometry>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <lidar_slam/lidar_slam.h>
#include <pcl/filters/voxel_grid.h>
#include <chrono>
#include <future>
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
        transNoise(i, i) = std::pow(params_.gicp_translation_noise, 2);

    Eigen::Matrix3d rotNoise = Eigen::Matrix3d::Zero();
    for (int i = 0; i < 3; ++i)
        rotNoise(i, i) = std::pow(params_.gicp_rotation_noise, 2);

    icp_information_ = Eigen::Matrix<double, 6, 6>::Zero();
    icp_information_.block<3, 3>(0, 0) = transNoise.inverse();
    icp_information_.block<3, 3>(3, 3) = rotNoise.inverse();

    if (params_.automatic_start)
    {
        Start();
    }
}


void LidarSlam::OdometryThread()
{
    LidarSlam::PointCloud tmp_cloud1 {};
    Eigen::Isometry3d accumulated_odometry = Eigen::Isometry3d::Identity();
    PointCloudPtr previous_cloud {};

    fast_gicp::FastVGICP<Point, Point> gicp;

    gicp.setResolution(params_.gicp_resolution);
    gicp.setVoxelAccumulationMode(fast_gicp::VoxelAccumulationMode::ADDITIVE_WEIGHTED);
    gicp.setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT1);
    gicp.setRegularizationMethod(fast_gicp::RegularizationMethod::FROBENIUS);

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

        if (!latest_cloud_local || (latest_cloud_local->size() < params_.minimal_cloud_size))
        {
            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(5));
            continue;
        }

        // initialize previous_cloud_ as well as first vertex whenever non-empty cloud arrives
        if (!previous_cloud)
        {
            previous_cloud = latest_cloud_local;
            gicp.setInputTarget(previous_cloud);
            last_odometry_vertex = new CloudVertexSE3(previous_cloud);
            last_odometry_vertex->setId(vertice_id_++);
            last_odometry_vertex->setToOrigin();
            last_odometry_vertex->setFixed(true);
            last_odometry_vertex->setMarginalized(true);

            if (odometry_callback_)
            {
                odometry_callback_(accumulated_odometry, latest_cloud_local->header.stamp);
            }
            continue;
        }

        gicp.setInputSource(latest_cloud_local);
        gicp.align(tmp_cloud1);
        if (gicp.hasConverged())
        {
            const Eigen::Matrix4d result = gicp.getFinalTransformation().cast<double>();
            const Eigen::Isometry3d isometry(result);

            SLAM_LOG << "ICP converged to:" << std::endl;
            SLAM_LOG << result << std::endl;

            if (odometry_callback_)
            {
                odometry_callback_(accumulated_odometry * isometry, latest_cloud_local->header.stamp);
            }

            auto shiftangle = GetAbsoluteShiftAngle(result);

            if (shiftangle.first > params_.new_node_min_translation || shiftangle.second > params_.new_node_after_rotation)
            {
                // when translation & rotation are small, we don't add this cloud into the graph,
                // neither change previous_cloud_ and accumulated_odometry_

                accumulated_odometry = accumulated_odometry * isometry;
                previous_cloud = latest_cloud_local;

                // we also add Vertex into the Graph
                auto vertex = new CloudVertexSE3(previous_cloud);
                vertex->setId(vertice_id_++);
                vertex->setFixed(false);
                vertex->setMarginalized(true);
                vertex->setEstimate(accumulated_odometry);

                auto edge = new g2o::EdgeSE3();
                edge->setId(edge_id_++);
                edge->setVertex(0, last_odometry_vertex);
                edge->setVertex(1, vertex);
                Eigen::Isometry3d t(result);
                edge->setMeasurement(t);
                edge->setInformation(icp_information_);


                // Store these guys in order not to loose them
                //gicp.swapSourceCovariances(last_odometry_vertex->covariances);
                //last_odometry_vertex->kdtree = gicp.sourceKdTree();
                {
                    const std::lock_guard<std::mutex> guard(vertices_mutex_);
                    vertices_.push_back(last_odometry_vertex);
                    odometryEdges_.push_back(edge);
                }
                gicp.swapSourceAndTarget();  // keep pre-calculated covariances/kd-tree

                last_odometry_vertex = vertex;
            }
        }
        else
        {
            SLAM_LOG << "ICP haven't converged" << std::endl;
        }
    }
    SLAM_LOG << "LidarSlam::OdometryThread() stopped!" << std::endl;
}

void LidarSlam::MappingThread()
{
    SLAM_LOG << "LidarSlam::MappingThread() is running!" << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    LidarSlam::PointCloud tmp_cloud1 {};
    fast_gicp::FastVGICP<Point, Point> gicp;
    gicp.setResolution(params_.gicp_resolution);
    gicp.setVoxelAccumulationMode(fast_gicp::VoxelAccumulationMode::ADDITIVE_WEIGHTED);
    gicp.setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT1);
    gicp.setRegularizationMethod(fast_gicp::RegularizationMethod::FROBENIUS);

    std::size_t vertices_amount = 0U;
    std::size_t odometry_edges_amount = 0U;
    std::size_t loop_edges_amount = 0U;
    bool latest_mapping_already_published = false;

    auto linearSolver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    linearSolver->setBlockOrdering(false);
    auto blockSolver = g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver));
    auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));

    /// Graph Optimizer
    g2o::SparseOptimizer optimizer_;
    optimizer_.setAlgorithm(solver);
    optimizer_.setVerbose(params_.optimizer_verbose);


    std::future<boost::optional<Eigen::Isometry3d>> loop_clousure_future {};
    CloudVertexSE3* loop_clousure_vertex0 {nullptr};
    CloudVertexSE3* loop_clousure_vertex1 {nullptr};
    std::chrono::milliseconds span (1);

    CloudVertexSE3* prev_vertex = nullptr;
    Eigen::Isometry3d prev_odometry{};

    while (mapping_thread_running_)
    {
        std::vector<CloudVertexSE3*> vertices{};
        std::vector<g2o::EdgeSE3*> odometryEdges{};
        {
            const std::lock_guard<std::mutex> guard(vertices_mutex_);
            vertices.swap(vertices_);
            odometryEdges.swap(odometryEdges_);
        }

        if (!vertices.empty())
        {
            for (auto vertex : vertices)
            {
                SLAM_LOG << "LidarSlam::MappingThread() vertex arrived" << std::endl;
                vertex->setMarginalized(false);
                optimizer_.addVertex(vertex);
                prev_vertex = vertex;
                prev_odometry = vertex->estimate();
                latest_mapping_already_published = false;
                vertices_amount ++;
            }
        }

        if (!odometryEdges.empty())
        {
            for (auto edge : odometryEdges)
            {
                SLAM_LOG << "LidarSlam::MappingThread() edge arrived" << std::endl;
                optimizer_.addEdge(edge);
                odometry_edges_amount++;
            }
        }

        if(prev_vertex == nullptr)
        {
            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(5));
            continue;
        }

        if(loop_clousure_future.valid() && loop_clousure_future.wait_for(span) == std::future_status::ready)
        {
            // loop-clousure result is here!
            auto result = loop_clousure_future.get();
            if(result.has_value())
            {
                g2o::EdgeSE3* edge = new g2o::EdgeSE3();
                edge->setVertex(0, loop_clousure_vertex0);
                edge->setVertex(1, loop_clousure_vertex1);
                edge->setMeasurement(result.value());
                optimizer_.addEdge(edge);
                loop_clousure_vertex0 = nullptr;
                loop_clousure_vertex1 = nullptr;
            }
        }

        // try to find new loop-clousure
        if(!loop_clousure_future.valid())
        {
            // we want no more than 20% of loop-closure edges
            if(vertices_amount > 10 && loop_edges_amount * 5 < odometry_edges_amount)
            {
                loop_clousure_vertex0 = prev_vertex;
                std::uniform_int_distribution<int> rand(0, vertice_id_ - 4);
                loop_clousure_vertex1 = reinterpret_cast<CloudVertexSE3*>(optimizer_.vertex(rand(gen)));

                loop_clousure_future = std::async([&]{
                  boost::optional<Eigen::Isometry3d> result;

                  gicp.setInputTarget(loop_clousure_vertex0->cloud);
                  //gicp.setInputTarget(loop_clousure_vertex0->cloud, loop_clousure_vertex0->kdtree);
                  //gicp.swapTargetCovariances(loop_clousure_vertex0->covariances);

                  gicp.setInputSource(loop_clousure_vertex1->cloud);
                  //gicp.setInputSource(loop_clousure_vertex1->cloud, loop_clousure_vertex1->kdtree);
                  //gicp.swapSourceCovariances(loop_clousure_vertex1->covariances);
                  gicp.align(tmp_cloud1);

                  // return covariance back to corresponsing vertices
                  //gicp.swapTargetCovariances(loop_clousure_vertex0->covariances);
                  //gicp.swapSourceCovariances(loop_clousure_vertex1->covariances);

                  if (gicp.hasConverged())
                  {
                      const Eigen::Matrix4d transform = gicp.getFinalTransformation().cast<double>();
                      result = Eigen::Isometry3d(transform);
                  }
                  return result;
                  });
            }
        }

        //SLAM_LOG << "LidarSlam::MappingThread() optimizer_.initializeOptimization()" << std::endl;
        if (optimizer_.initializeOptimization())
        {
            //optimizer_.computeActiveErrors();
            //SLAM_LOG << "LidarSlam::MappingThread() optimizer_.optimize()" << std::endl;
            if (optimizer_.optimize(5) > 0)
            {
                if (mapping_callback_ && !latest_mapping_already_published)
                {
                    //SLAM_LOG << "LidarSlam::MappingThread() sending mapping_callback_" << std::endl;
                    const Eigen::Isometry3d p = prev_vertex->estimate() * prev_odometry.inverse();
                    mapping_callback_(p, prev_vertex->cloud->header.stamp);
                    latest_mapping_already_published = true;
                }
            }
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(50));
        }
    }
    optimizer_.clear();
    bool stop = true;
    optimizer_.setForceStopFlag(&stop);

    SLAM_LOG << "LidarSlam::MappingThread() stopped!" << std::endl;
}

void LidarSlam::Start()
{
    if (!mapping_thread_running_ && !mapping_thread_.joinable())
    {
        mapping_thread_running_ = true;
        mapping_thread_ = std::thread(&LidarSlam::MappingThread, this);
    }

    if (!odometry_thread_running_ && !odometry_thread_.joinable())
    {
        odometry_thread_running_ = true;
        odometry_thread_ = std::thread(&LidarSlam::OdometryThread, this);
    }
//    if (loopclousure_thread_running_ && !loopclousure_thread_.joinable())
//    {
//        loopclousure_thread_running_ = true;
//        loopclousure_thread_ = std::thread(&LidarSlam::LoopClousureThread, this);
//    }
}

void LidarSlam::Stop()
{
    odometry_thread_running_ = false;
    mapping_thread_running_ = false;
    //loopclousure_thread_running_ = false;
    if (odometry_thread_.joinable())
    {
        odometry_thread_.join();
    }
    if (mapping_thread_.joinable())
    {
        mapping_thread_.join();
    }
//    if (loopclousure_thread_.joinable())
//    {
//        loopclousure_thread_.join();
//    }
}

LidarSlam::~LidarSlam()
{
    Stop();
}

void LidarSlam::AddPointCloud(const PointCloudPtr& msg)
{
    if (!msg)
    {
        throw std::runtime_error("LidarSlam::AddPointCloud() uninitialized cloud given");
    }
    if (latest_cloud_ && msg->header.stamp <= latest_cloud_->header.stamp)
    {
        throw std::runtime_error("LidarSlam::AddPointCloud() received clouds must have increasing timestamps");
    }
    {
        const std::lock_guard<std::mutex> guard(latest_cloud_mutex_);
        latest_cloud_ = msg;
    }
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

    return std::make_pair(translation, rotation * 2.);
}

}  // namespace lidar_slam