#include <Eigen/Geometry>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <lidar_slam/helpers.h>
#include <lidar_slam/lidar_slam.h>
#include <chrono>
#include <future>
#include <mutex>
#include <random>
#include <thread>

// following includes come last
#include <fast_gicp/gicp/fast_vgicp.hpp>
#include <fast_gicp/gicp/impl/fast_gicp_impl.hpp>
#include <fast_gicp/gicp/impl/fast_vgicp_impl.hpp>
#include <fast_gicp/gicp/impl/lsq_registration_impl.hpp>

#ifdef USE_SLAM_LOGGING
#define SLAM_LOG std::cout
#else
#define SLAM_LOG \
    if (false)   \
    std::cout
#endif

using Point = lidar_slam::LidarSlam::Point;
template class fast_gicp::FastVGICP<Point, Point>;
template class fast_gicp::FastGICP<Point, Point>;
template class fast_gicp::LsqRegistration<Point, Point>;

namespace lidar_slam
{

struct LidarSlam::Impl
{
    Impl(const LidarSlamParameters& params);

    CallbackType odometry_callback_{}, mapping_callback_{};

    /// Cloud updated once AddPointCloud() is called
    PointCloudPtr latest_cloud_{};
    /// Corresponding mutex
    std::mutex latest_cloud_mutex_{};

    /// @brief List of vertices, waiting to be added into the Graph
    /// Every such vertex must be marginalized, in order not to ruin currently running optimization
    /// Mapping thread will set them correct when adding into the Graph
    std::vector<g2o::VertexSE3*> vertices_{};
    std::vector<g2o::EdgeSE3*> odometryEdges_{};
    /// Mutex, controlling both @ref vertices_ and @ref odometryEdges_
    std::mutex vertices_mutex_{};

    /// Threads for independend SLAM parts
    std::thread odometry_thread_{}, mapping_thread_{};
    /// Respective bool atomics
    std::atomic_bool odometry_thread_running_{};
    std::atomic_bool mapping_thread_running_{};

    /// Main loop-functions for respective threads
    void OdometryThread();
    void MappingThread();

    /// read-only reference to parameters
    /// SLAM may not change it, but it still could be changed from outside
    LidarSlamParameters params_;

    std::atomic<int> vertex_id_{}, edge_id_{};

    /// Information matrix (i.e. inverse covariance of the 6DoF measurement: X Y Z QX QY QZ)
    Eigen::Matrix<double, 6, 6> icp_information_{};

    std::unordered_map<int, CloudConstPtr> clouds_{};
    std::mutex clouds_mutex_{};

    fast_gicp::FastVGICP<Point, Point> gicp1;
    fast_gicp::FastVGICP<Point, Point> gicp2;

    boost::optional<Eigen::Matrix4d> AlignCloud(PointCloud::Ptr source);
};

LidarSlam::Impl::Impl(const LidarSlamParameters& params) : params_(params)
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

    gicp1.setResolution(params_.gicp_resolution);
    gicp1.setVoxelAccumulationMode(fast_gicp::VoxelAccumulationMode::ADDITIVE_WEIGHTED);
    gicp1.setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT1);
    gicp1.setRegularizationMethod(fast_gicp::RegularizationMethod::FROBENIUS);

    gicp2.setResolution(params_.gicp_resolution);
    gicp2.setVoxelAccumulationMode(fast_gicp::VoxelAccumulationMode::ADDITIVE_WEIGHTED);
    gicp2.setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT1);
    gicp2.setRegularizationMethod(fast_gicp::RegularizationMethod::FROBENIUS);
}

LidarSlam::LidarSlam(const LidarSlamParameters& params) : impl_(new Impl(params))
{

    if (impl_->params_.automatic_start)
    {
        Start();
    }
}

void LidarSlam::SetOdometryCallback(CallbackType&& callback)
{
    impl_->odometry_callback_ = callback;
}

/// Set-up callback which publishes high-latency transform, suitable for mapping
/// returns transform from odometry frame into mapping frame
void LidarSlam::SetMappingCallback(CallbackType&& callback)
{
    impl_->mapping_callback_ = callback;
}

boost::optional<Eigen::Matrix4d> LidarSlam::Impl::AlignCloud(PointCloud::Ptr source)
{
    boost::optional<Eigen::Matrix4d> result;
    if (params_.gicp_use_l2r_check)
    {
    }
    else
    {
        LidarSlam::PointCloud tmp_cloud1{};
        gicp1.setInputSource(source);
        gicp1.align(tmp_cloud1);
        if (gicp1.hasConverged())
        {
            result = gicp1.getFinalTransformation().cast<double>();
        }
    }
    return result;
}

void LidarSlam::Impl::OdometryThread()
{

    Eigen::Isometry3d accumulated_odometry = Eigen::Isometry3d::Identity();
    PointCloudPtr previous_cloud{};

    // Adding vertex to the edge (edge->setVertex(..)) changes vertex too, which is not thread-safe
    // So we cannot add last_odometry_vertex_ directly to the Graph, but rather wait for the next odometry step
    // During this time we keep "current" odometry vertex here
    g2o::VertexSE3* last_odometry_vertex{nullptr};

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
            gicp1.setInputTarget(previous_cloud);
            last_odometry_vertex = new g2o::VertexSE3();
            last_odometry_vertex->setId(vertex_id_++);
            last_odometry_vertex->setToOrigin();
            last_odometry_vertex->setFixed(true);
            last_odometry_vertex->setMarginalized(true);
            {
                const std::lock_guard<std::mutex> guard(clouds_mutex_);
                clouds_[last_odometry_vertex->id()] = CloudConstPtr(previous_cloud);
            }

            if (odometry_callback_)
            {
                odometry_callback_(accumulated_odometry, latest_cloud_local->header.stamp);
            }
            continue;
        }

        // gicp1.setInputSource(latest_cloud_local);
        // gicp1.align(tmp_cloud1);
        auto result = AlignCloud(latest_cloud_local);
        if (result.has_value())  //(gicp1.hasConverged())
        {
            // const Eigen::Matrix4d result = gicp1.getFinalTransformation().cast<double>();
            const Eigen::Isometry3d isometry(result.value());

            SLAM_LOG << "ICP converged" << std::endl << result.value() << std::endl;

            if (odometry_callback_)
            {
                odometry_callback_(accumulated_odometry * isometry, latest_cloud_local->header.stamp);
            }

            auto shiftangle = Helpers::GetAbsoluteShiftAngle(result.value());

            if (shiftangle.first > params_.new_node_min_translation ||
                shiftangle.second > params_.new_node_after_rotation)
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
                    clouds_[vertex->id()] = CloudConstPtr(previous_cloud);
                }

                auto edge = new g2o::EdgeSE3();
                edge->setId(edge_id_++);
                edge->setVertex(0, last_odometry_vertex);
                edge->setVertex(1, vertex);
                edge->setMeasurement(isometry);
                edge->setInformation(icp_information_);

                // Store these guys in order not to loose them
                // gicp.swapSourceCovariances(last_odometry_vertex->covariances);
                // last_odometry_vertex->kdtree = gicp.sourceKdTree();
                {
                    const std::lock_guard<std::mutex> guard(vertices_mutex_);
                    vertices_.push_back(last_odometry_vertex);
                    odometryEdges_.push_back(edge);
                }
                gicp1.swapSourceAndTarget();  // keep pre-calculated covariances/kd-tree

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

void LidarSlam::Impl::MappingThread()
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
        g2o::VertexSE3* vertex{nullptr};
        Eigen::Isometry3d odometry{};
        CloudConstPtr cloud;
        // avoids sending same-frame mapping several times
        std::atomic<bool> mapping_already_published{false};
    };

    VertexInfo last_mapping_vertex{};
    std::atomic<int> latest_loop_closure_vertex_id{};

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

        if (last_mapping_vertex.vertex == nullptr)
        {
            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(5));
            continue;
        }

        // process loop-clousures
        if (loop_clousure_future.valid())
        {
            if (loop_clousure_future.wait_for(span) == std::future_status::ready)
            {
                // loop-clousure result is here!
                auto result = loop_clousure_future.get();
                if (result.result.has_value() && result.loop_clousure_vertex0 != nullptr &&
                    result.loop_clousure_vertex1 != nullptr)
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
                    loop_edges_amount++;
                }
            }
        }
        else
        {
            const bool enough_vertices = last_mapping_vertex.id > params_.loop_closure_min_vertices;
            // we want no more than some % of loop-closure edges
            const bool not_too_many_loop_closures = loop_edges_amount * params_.loop_closure_edge_rato < edge_id_;
            const bool was_not_looped_yet = last_mapping_vertex.id != latest_loop_closure_vertex_id;

            if (enough_vertices && not_too_many_loop_closures && was_not_looped_yet)
            {

                LoopClosureSearchResult future_result{};
                future_result.loop_clousure_vertex0 = last_mapping_vertex.vertex;
                latest_loop_closure_vertex_id = last_mapping_vertex.id;

                std::uniform_int_distribution<int> rand(0, last_mapping_vertex.id - 1);
                int other_id = rand(gen);
                while (other_id == last_mapping_vertex.id)
                {
                    other_id = rand(gen);
                }

                SLAM_LOG << "Trying to find Loop Closure (between Vertex# " << last_mapping_vertex.id << " and Vertex#"
                         << other_id << ")" << std::endl;

                future_result.loop_clousure_vertex1 = (g2o::VertexSE3*)optimizer_.vertex(other_id);

                loop_clousure_future = std::async([&]() {
                    LoopClosureSearchResult result{future_result.loop_clousure_vertex0,
                                                   future_result.loop_clousure_vertex1};
                    LidarSlam::PointCloud tmp_cloud1{};
                    // boost::optional<Eigen::Isometry3d> result;

                    gicp.setInputTarget(last_mapping_vertex.cloud);
                    // gicp.setInputTarget(loop_clousure_vertex0->cloud, loop_clousure_vertex0->kdtree);
                    // gicp.swapTargetCovariances(loop_clousure_vertex0->covariances);

                    gicp.setInputSource(clouds_.at(other_id));
                    // gicp.setInputSource(loop_clousure_vertex1->cloud, loop_clousure_vertex1->kdtree);
                    // gicp.swapSourceCovariances(loop_clousure_vertex1->covariances);
                    gicp.align(tmp_cloud1);

                    // return covariance back to corresponsing vertices
                    // gicp.swapTargetCovariances(loop_clousure_vertex0->covariances);
                    // gicp.swapSourceCovariances(loop_clousure_vertex1->covariances);

                    if (gicp.hasConverged())
                    {
                        const Eigen::Matrix4d transform = gicp.getFinalTransformation().cast<double>();
                        result.result = Eigen::Isometry3d(transform);
                    }
                    return result;
                });
            }
        }

        // SLAM_LOG << "LidarSlam::MappingThread() optimizer_.initializeOptimization()" << std::endl;
        if (optimizer_.initializeOptimization())
        {

            // SLAM_LOG << "LidarSlam::MappingThread() optimizer_.optimize()" << std::endl;
            if (optimizer_.optimize(5) > 0)
            {
                if (mapping_callback_ && latest_loop_closure_vertex_id != last_mapping_vertex.id)
                {
                    SLAM_LOG << "LidarSlam::MappingThread() sending mapping_callback_" << std::endl;

                    const Eigen::Isometry3d p =
                        last_mapping_vertex.vertex->estimate() * last_mapping_vertex.odometry.inverse();
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
    if (loop_clousure_future.valid())
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
    if (!impl_->mapping_thread_running_ && !impl_->mapping_thread_.joinable())
    {
        impl_->mapping_thread_running_ = true;
        impl_->mapping_thread_ = std::thread(&LidarSlam::Impl::MappingThread, impl_.get());
    }

    if (!impl_->odometry_thread_running_ && !impl_->odometry_thread_.joinable())
    {
        impl_->odometry_thread_running_ = true;
        impl_->odometry_thread_ = std::thread(&LidarSlam::Impl::OdometryThread, impl_.get());
    }
}

void LidarSlam::Stop()
{
    impl_->odometry_thread_running_ = false;
    impl_->mapping_thread_running_ = false;
    if (impl_->odometry_thread_.joinable())
    {
        impl_->odometry_thread_.join();
    }
    if (impl_->mapping_thread_.joinable())
    {
        impl_->mapping_thread_.join();
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
    if (impl_->latest_cloud_ && msg->header.stamp <= impl_->latest_cloud_->header.stamp)
    {
        throw std::runtime_error("LidarSlam::AddPointCloud() received clouds must have increasing timestamps");
    }
    {
        const std::lock_guard<std::mutex> guard(impl_->latest_cloud_mutex_);
        impl_->latest_cloud_ = msg;
    }
}

}  // namespace lidar_slam