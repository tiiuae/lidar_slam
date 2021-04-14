#include <chrono>
#include <future>
#include <random>
#include <thread>
#include <Eigen/Geometry>
#include <lidar_slam/lidar_slam.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <lidar_slam/helpers.h>

// following includes come last
#include <fast_gicp/gicp/fast_vgicp.hpp>
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
    g2o::VertexSE3* last_odometry_vertex {nullptr};

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
            last_odometry_vertex = new g2o::VertexSE3();
            last_odometry_vertex->setId(vertex_id_++);
            last_odometry_vertex->setToOrigin();
            last_odometry_vertex->setFixed(true);
            last_odometry_vertex->setMarginalized(true);
            {
                const std::lock_guard<std::mutex> guard(clouds_mutex_);
                clouds_[last_odometry_vertex->id()] = CloudConstPtr (previous_cloud);
            }

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

            SLAM_LOG << "ICP converged" << std::endl;
            //SLAM_LOG << result << std::endl;

            if (odometry_callback_)
            {
                odometry_callback_(accumulated_odometry * isometry, latest_cloud_local->header.stamp);
            }

            auto shiftangle = Helpers::GetAbsoluteShiftAngle(result);

            if (shiftangle.first > params_.new_node_min_translation || shiftangle.second > params_.new_node_after_rotation)
            {
                // when translation & rotation are small, we don't add this cloud into the graph,
                // neither change previous_cloud_ and accumulated_odometry_
                accumulated_odometry = accumulated_odometry * isometry;
                previous_cloud = latest_cloud_local;

                // we also add Vertex into the Graph
                auto vertex = new g2o::VertexSE3();
                vertex->setId(vertex_id_++);
                vertex->setFixed(false);
                vertex->setMarginalized(true);
                vertex->setEstimate(accumulated_odometry);

                {
                    const std::lock_guard<std::mutex> guard(clouds_mutex_);
                    clouds_[vertex->id()] = CloudConstPtr (previous_cloud);
                }

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

    std::random_device rd{};
    std::mt19937 gen(rd());

    fast_gicp::FastVGICP<Point, Point> gicp;
    gicp.setResolution(params_.gicp_resolution);
    gicp.setVoxelAccumulationMode(fast_gicp::VoxelAccumulationMode::ADDITIVE_WEIGHTED);
    gicp.setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT1);
    gicp.setRegularizationMethod(fast_gicp::RegularizationMethod::FROBENIUS);

    /// Graph Optimizer
    g2o::SparseOptimizer optimizer_;
    auto linearSolver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    linearSolver->setBlockOrdering(false);
    auto blockSolver = g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver));
    auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));
    optimizer_.setAlgorithm(solver);
    optimizer_.setVerbose(params_.optimizer_verbose);

    // local variables required for loop-closure
    struct LoopClosureSearchResult
    {
        g2o::VertexSE3* loop_clousure_vertex0{nullptr};
        g2o::VertexSE3* loop_clousure_vertex1{nullptr};
        boost::optional<Eigen::Isometry3d> result;
    };

    std::future<LoopClosureSearchResult> loop_clousure_future{};

    const std::chrono::milliseconds span(1);

    struct VertexInfo
    {
        int id;
        g2o::VertexSE3* vertex {nullptr};
        Eigen::Isometry3d odometry{};
        CloudConstPtr cloud;
        // avoids sending same-frame mapping several times
        std::atomic<bool> mapping_already_published {false};
    };

    VertexInfo last_mapping_vertex{};
    std::atomic<int> latest_loop_closure_vertex_id {};

    int loop_edges_amount = 0U;

    while (mapping_thread_running_)
    {
        std::vector<g2o::VertexSE3*> vertices{};
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
                last_mapping_vertex.vertex = vertex;
                last_mapping_vertex.id = vertex->id();
                last_mapping_vertex.odometry = vertex->estimate();
                last_mapping_vertex.mapping_already_published = false;
                const std::lock_guard<std::mutex> guard(clouds_mutex_);
                last_mapping_vertex.cloud = clouds_.at(last_mapping_vertex.id);
            }
        }

        if (!odometryEdges.empty())
        {
            for (auto edge : odometryEdges)
            {
                SLAM_LOG << "LidarSlam::MappingThread() edge arrived" << std::endl;
                optimizer_.addEdge(edge);
            }
        }

        if(last_mapping_vertex.vertex == nullptr)
        {
            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(5));
            continue;
        }

        // process loop-clousures
        if(loop_clousure_future.valid())
        {
            if(loop_clousure_future.wait_for(span) == std::future_status::ready)
            {
                // loop-clousure result is here!
                auto result = loop_clousure_future.get();
                if(result.result.has_value() && result.loop_clousure_vertex0 != nullptr && result.loop_clousure_vertex1 != nullptr)
                {
                    SLAM_LOG << "Loop Closure found!" << std::endl;
                    SLAM_LOG << result.result.value().matrix() << std::endl;
                    latest_loop_closure_vertex_id = result.loop_clousure_vertex0->id();

                    auto edge = new g2o::EdgeSE3();
                    edge->setId(edge_id_++);
                    edge->setVertex(0, result.loop_clousure_vertex0);
                    edge->setVertex(1, result.loop_clousure_vertex1);
                    edge->setMeasurement(result.result.value());
                    edge->setInformation(icp_information_);
                    optimizer_.addEdge(edge);
                    loop_edges_amount ++;
                }
            }
        }
        else
        {
            const bool enough_vertices = last_mapping_vertex.id > params_.loop_closure_min_vertices;
            // we want no more than some % of loop-closure edges
            const bool not_too_many_loop_closures = loop_edges_amount * params_.loop_closure_edge_rato < edge_id_;
            const bool was_not_looped_yet = last_mapping_vertex.id != latest_loop_closure_vertex_id;

            if(enough_vertices && not_too_many_loop_closures && was_not_looped_yet)
            {

                LoopClosureSearchResult future_result{};
                future_result.loop_clousure_vertex0 = last_mapping_vertex.vertex;
                latest_loop_closure_vertex_id = last_mapping_vertex.id;

                std::uniform_int_distribution<int> rand(0, last_mapping_vertex.id - 1);
                int other_id = rand(gen);
                while(other_id == last_mapping_vertex.id)
                {
                    other_id = rand(gen);
                }

                SLAM_LOG << "Trying to find Loop Closure (between Vertex# " << last_mapping_vertex.id << " and Vertex#" << other_id << ")" << std::endl;


                future_result.loop_clousure_vertex1 = (g2o::VertexSE3*)optimizer_.vertex(other_id);


                loop_clousure_future = std::async([&](){
                  LoopClosureSearchResult result{future_result.loop_clousure_vertex0, future_result.loop_clousure_vertex1};
                  LidarSlam::PointCloud tmp_cloud1 {};
                  //boost::optional<Eigen::Isometry3d> result;

                  gicp.setInputTarget(last_mapping_vertex.cloud);
                  //gicp.setInputTarget(loop_clousure_vertex0->cloud, loop_clousure_vertex0->kdtree);
                  //gicp.swapTargetCovariances(loop_clousure_vertex0->covariances);

                  gicp.setInputSource(clouds_.at(other_id));
                  //gicp.setInputSource(loop_clousure_vertex1->cloud, loop_clousure_vertex1->kdtree);
                  //gicp.swapSourceCovariances(loop_clousure_vertex1->covariances);
                  gicp.align(tmp_cloud1);

                  // return covariance back to corresponsing vertices
                  //gicp.swapTargetCovariances(loop_clousure_vertex0->covariances);
                  //gicp.swapSourceCovariances(loop_clousure_vertex1->covariances);

                  if (gicp.hasConverged())
                  {
                      const Eigen::Matrix4d transform = gicp.getFinalTransformation().cast<double>();
                      result.result = Eigen::Isometry3d(transform);
                  }
                  return result;
                  });

            }
        }

        //SLAM_LOG << "LidarSlam::MappingThread() optimizer_.initializeOptimization()" << std::endl;
        if (optimizer_.initializeOptimization())
        {

            //SLAM_LOG << "LidarSlam::MappingThread() optimizer_.optimize()" << std::endl;
            if (optimizer_.optimize(5) > 0)
            {
                if (mapping_callback_ && latest_loop_closure_vertex_id != last_mapping_vertex.id)
                {
                    SLAM_LOG << "LidarSlam::MappingThread() sending mapping_callback_" << std::endl;

                    const Eigen::Isometry3d p = last_mapping_vertex.vertex->estimate() * last_mapping_vertex.odometry.inverse();
                    mapping_callback_(p, last_mapping_vertex.cloud->header.stamp);
                    latest_loop_closure_vertex_id = last_mapping_vertex.id;
                }
            }
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(50));
        }
    }
    // make sure std::future is complete before exit
    if(loop_clousure_future.valid())
    {
        loop_clousure_future.get();
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
}

void LidarSlam::Stop()
{
    odometry_thread_running_ = false;
    mapping_thread_running_ = false;
    if (odometry_thread_.joinable())
    {
        odometry_thread_.join();
    }
    if (mapping_thread_.joinable())
    {
        mapping_thread_.join();
    }
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




}  // namespace lidar_slam