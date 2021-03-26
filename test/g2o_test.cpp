/// Tests minimal functional code, resembling 3D SLAM

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/types/slam3d/edge_se3.h"
#include "gtest/gtest.h"

using namespace std;
using namespace g2o;

typedef g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType> SlamLinearSolver;

/// Test class to validate expected g2o behaviour
class G2OTest : public ::testing::Test
{
  public:
    G2OTest()
    {
        auto linearSolver = g2o::make_unique<SlamLinearSolver>();
        linearSolver->setBlockOrdering(false);
        auto blockSolver = g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver));
        auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));
        optimizer_.setAlgorithm(solver);
    }

  protected:
    g2o::SparseOptimizer optimizer_;
};


/// Initial estimate for vertex1 is set "randomly", but edge has identity constraint
/// => after optimization vertex1 shall go to origin
TEST_F(G2OTest, BasicTest)
{
    g2o::VertexSE3* vertex0 = new g2o::VertexSE3();
    vertex0->setId(0);
    vertex0->setToOrigin();
    vertex0->setFixed(true);
    optimizer_.addVertex(vertex0);

    g2o::VertexSE3* vertex1 = new g2o::VertexSE3();
    vertex1->setId(1);
    g2o::Isometry3 p2 = g2o::Isometry3::Identity();
    p2.translation() << 3., 5., 7.;
    vertex1->setEstimate(p2);
    vertex1->setFixed(false);
    optimizer_.addVertex(vertex1);

    g2o::EdgeSE3* edge = new g2o::EdgeSE3();
    edge->setInformation(g2o::EdgeSE3::InformationType::Identity());
    edge->setMeasurement(g2o::Isometry3::Identity());
    edge->setVertex(0, vertex0);
    edge->setVertex(1, vertex1);
    optimizer_.addEdge(edge);

    optimizer_.initializeOptimization();
    optimizer_.computeActiveErrors();
    EXPECT_LT(0., optimizer_.chi2());
    int numOptimization = optimizer_.optimize(30);
    EXPECT_LT(1, numOptimization);
    EXPECT_GT(1e-6, optimizer_.chi2());
    EXPECT_GT(1e-6, optimizer_.activeChi2());

    g2o::VertexSE3* v2AfterOpti = dynamic_cast<g2o::VertexSE3*>(optimizer_.vertex(1));
    EXPECT_DOUBLE_EQ(0., (v2AfterOpti->estimate().translation() - g2o::Vector3::Zero()).norm());
    EXPECT_DOUBLE_EQ(0., (v2AfterOpti->estimate().rotation().diagonal() - g2o::Vector3::Ones()).norm());
}


/// Initial estimate for vertex1 is set "randomly", but edge has different constraint
/// => after optimization vertex1 shall go to pose, defined by edge
TEST_F(G2OTest, BasicTest2)
{
    g2o::VertexSE3* vertex0 = new g2o::VertexSE3();
    vertex0->setId(0);
    vertex0->setToOrigin();
    vertex0->setFixed(true);
    optimizer_.addVertex(vertex0);

    g2o::VertexSE3* vertex1 = new g2o::VertexSE3();
    vertex1->setId(1);
    g2o::Isometry3 p1 = g2o::Isometry3::Identity();
    p1.translation() << 11., 17., 23.;
    vertex1->setEstimate(p1);
    vertex1->setFixed(false);
    optimizer_.addVertex(vertex1);

    g2o::Isometry3 p2 = g2o::Isometry3::Identity();
    p2.rotate(Eigen::Quaterniond(0.1, 0., 0., 1.) );
    p2.translation() << 3., 5., 7.;

    g2o::EdgeSE3* edge = new g2o::EdgeSE3();
    edge->setInformation(g2o::EdgeSE3::InformationType::Identity());
    edge->setMeasurement(p2);
    edge->setVertex(0, vertex0);
    edge->setVertex(1, vertex1);
    optimizer_.addEdge(edge);

    optimizer_.initializeOptimization();
    optimizer_.computeActiveErrors();
    EXPECT_LT(0., optimizer_.chi2());
    int numOptimization = optimizer_.optimize(30);
    EXPECT_LT(1, numOptimization);
//    EXPECT_GT(1e-6, optimizer_.chi2());
//    EXPECT_GT(1e-6, optimizer_.activeChi2());

    g2o::VertexSE3* v2AfterOpti = dynamic_cast<g2o::VertexSE3*>(optimizer_.vertex(1));
    EXPECT_GT(1e-6, (v2AfterOpti->estimate().translation() - p2.translation()).norm());
    //EXPECT_DOUBLE_EQ(0., (v2AfterOpti->estimate().rotation().diagonal() - g2o::Vector3::Ones()).norm());
}

