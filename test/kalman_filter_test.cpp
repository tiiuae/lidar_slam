#include <gtest/gtest.h>
#include <lidar_slam/kalman_filter.h>

using namespace lidar_slam;
using namespace Eigen;

class KalmanFilterTest : public ::testing::Test
{
  public:
    KalmanFilterTest()
        :

          filter_(KalmanFilter::OriginInitialState,
                  KalmanFilter::SmallInitialVariance,
                  KalmanFilter::DefaultPredictionProcessNoise,
                  1.F)
    {
    }

  protected:
    KalmanFilter filter_;
};

/// Basic Prediction test
TEST_F(KalmanFilterTest, BasicPredictTest)
{
    EXPECT_EQ(0.F, filter_.state().Position.norm());
    EXPECT_EQ(1.F, filter_.state().Quaternion.norm());
    EXPECT_EQ(1.F, filter_.state().Quaternion[3]);
    EXPECT_EQ(0.F, filter_.state().Velocity.norm());
    EXPECT_EQ(0.F, filter_.state().AngularVelocity.norm());

    filter_.Predict(2.F);
    EXPECT_EQ(0.F, filter_.state().Position.norm());
    EXPECT_EQ(1.F, filter_.state().Quaternion.norm());
    EXPECT_EQ(1.F, filter_.state().Quaternion[3]);
    EXPECT_EQ(0.F, filter_.state().Velocity.norm());
    EXPECT_EQ(0.F, filter_.state().AngularVelocity.norm());
}

/// Basic Update test
TEST_F(KalmanFilterTest, BasicUpdateTest)
{
    Isometry3f observation = Isometry3f::Identity();
    observation.pretranslate(Vector3f(1, 0, 0));
    filter_.AbsoluteUpdate(observation, 2.F);

    const float x = filter_.state().Position[0];
    const float dx = filter_.state().Velocity[0];

    EXPECT_TRUE(1.F > x && x > 0.F) << "x=" << x;
    EXPECT_EQ(0.F, filter_.state().Position[1]);
    EXPECT_EQ(0.F, filter_.state().Position[2]);
    EXPECT_TRUE(1.F > dx && dx > 0.F) << "dx=" << dx;
    EXPECT_EQ(0.F, filter_.state().Velocity[1]);
    EXPECT_EQ(0.F, filter_.state().Velocity[2]);
    EXPECT_EQ(1.F, filter_.state().Quaternion.norm());
    EXPECT_EQ(1.F, filter_.state().Quaternion[3]);
    EXPECT_EQ(0.F, filter_.state().AngularVelocity.norm());
}

/// Basic Rotation test
TEST_F(KalmanFilterTest, BasicRotationTest)
{
    Isometry3f observation = Isometry3f::Identity();
    observation.pretranslate(Vector3f(1, 0, 0));
    observation.prerotate(AngleAxisf(1.F, Vector3f::UnitX()));
    filter_.AbsoluteUpdate(observation, 2.F);

    const float x = filter_.state().Position[0];
    const float dx = filter_.state().Velocity[0];
    const float qx = filter_.state().Quaternion[0];
    const float wx = filter_.state().AngularVelocity[0];

    EXPECT_LT(1.F, filter_.state().Quaternion.norm());
    EXPECT_EQ(1.F, filter_.state().Quaternion[3]);
    EXPECT_TRUE(1.F > qx && qx > 0.F) << "qx=" << qx;
    EXPECT_TRUE(1.F > wx && wx > 0.F) << "wx=" << wx;

    EXPECT_EQ(0.F, filter_.state().Quaternion[1]);
    EXPECT_EQ(0.F, filter_.state().Quaternion[2]);
    EXPECT_EQ(0.F, filter_.state().AngularVelocity[1]);
    EXPECT_EQ(0.F, filter_.state().AngularVelocity[2]);

    filter_.Predict(3.F);

    const float x2 = filter_.state().Position[0];
    const float dx2 = filter_.state().Velocity[0];
    const float ax2 = filter_.state().Quaternion[0];
    const float wx2 = filter_.state().AngularVelocity[0];

    // Since velocities are positive, we shall witness increase in x and ax
    EXPECT_TRUE(2.F > x2 && x2 > x) << "x=" << x << ", x2=" << x2;
    EXPECT_TRUE(dx2 == dx) << "dx=" << dx << ", dx2=" << dx2;
    EXPECT_TRUE(1.F > ax2 && ax2 > qx) << "ax=" << qx << ", ax2=" << x2;
    EXPECT_TRUE(wx2 == wx) << "wx=" << wx << ", wx2=" << x2;
}

/// If we "Update" with the same absolute pose several times, filter will converge to it, with velocity~=0
TEST_F(KalmanFilterTest, BasicConvergenceTest)
{
    Isometry3f observation = Isometry3f::Identity();
    observation.translate(Vector3f(1, 2, 3));
    Quaternion<float> true_rotation(1, 1.F, 0.5F, 0.25F);
    true_rotation.normalize();
    observation.rotate(true_rotation);

    for(int i = 0; i < 10; i++)
    {
        filter_.AbsoluteUpdate(observation, i + 2.F, 0.001);
    }
    //std::cout << "Resulting State: " << filter_.state().AllValues.transpose() << std::endl;

    EXPECT_NEAR(1.F, filter_.state().Position[0], 1e-5F);
    EXPECT_NEAR(2.F, filter_.state().Position[1], 1e-5F);
    EXPECT_NEAR(3.F, filter_.state().Position[2], 1e-5F);

    EXPECT_NEAR(0.F, filter_.state().Velocity[0], 1e-5F);
    EXPECT_NEAR(0.F, filter_.state().Velocity[1], 1e-5F);
    EXPECT_NEAR(0.F, filter_.state().Velocity[2], 1e-5F);

    EXPECT_NEAR(0.F, filter_.state().AngularVelocity[0], 1e-5F);
    EXPECT_NEAR(0.F, filter_.state().AngularVelocity[1], 1e-5F);
    EXPECT_NEAR(0.F, filter_.state().AngularVelocity[2], 1e-5F);

    // resulting quaternion components (non-unit-length)
    const float qx = filter_.state().Quaternion[0];
    const float qy = filter_.state().Quaternion[1];
    const float qz = filter_.state().Quaternion[2];
    const float qw = filter_.state().Quaternion[3];

    EXPECT_NEAR(qx/qw, 1.F, 1e-5F);
    EXPECT_NEAR(qy/qw, 0.5F, 1e-5F);
    EXPECT_NEAR(qz/qw, 0.25F, 1e-5F);
}

TEST_F(KalmanFilterTest, VelocityEstimationTest)
{
    const Vector3f true_translation(1.F, 0.5F, 0.F);
    Isometry3f accumulated_pose = Isometry3f::Identity();

    // Initial speed is unknown
    KalmanFilter::StateVector InitialVariance = 0.01F * KalmanFilter::StateVector::Ones();
    InitialVariance[3] = 1.F;
    InitialVariance[4] = 1.F;
    InitialVariance[5] = 1.F;

    filter_.Reset(KalmanFilter::OriginInitialState,
                  InitialVariance,
                  KalmanFilter::DefaultPredictionProcessNoise,
                  1.F);

    for(int i = 0; i < 20; i++)
    {
        accumulated_pose.translate(true_translation);
        filter_.Predict(i + 2.F);
        std::cout << "Predict State: " << filter_.state().AllValues.transpose() << std::endl;
        filter_.AbsoluteUpdate(accumulated_pose, i + 2.F, 0.01F);
        std::cout << "Update State: " << filter_.state().AllValues.transpose() << std::endl;
        //std::cout << "Real Pose: " << accumulated_pose.translation().transpose() << std::endl;
    }
    //std::cout << "Final Resulting State: " << filter_.state().AllValues.transpose() << std::endl;
    //std::cout << "Final Resulting Covariance: " << std::endl << filter_.covariance() << std::endl;


}
//
//
//TEST_F(KalmanFilterTest, IncrementalUpdateTest)
//{
//    //Isometry3f observation = Isometry3f::Identity();
//    //observation.translate(Vector3f(1.F, 0.5F, 0.F));
//    const Vector3f true_translation(1.F, 0.5F, 0.F);
//    Quaternion<float> true_rotation(0.9F, 0.F, 0.F, 0.1F);
//    true_rotation.normalize();
//    //observation.rotate(true_rotation);
//    Isometry3f accumulated_pose = Isometry3f::Identity();
//
//    for(int i = 0; i < 10; i++)
//    {
//        accumulated_pose.prerotate(true_rotation);
//        accumulated_pose.translate(true_translation);
//        //accumulated_pose = accumulated_pose * observation;
//
//        filter_.AbsoluteUpdate(accumulated_pose, i + 2.F, 0.001);
//        std::cout << "State: " << filter_.state().AllValues.transpose() << std::endl;
//        std::cout << "Pose: " << accumulated_pose.translation().transpose() << ", ";
//        std::cout << Eigen::Quaternionf(accumulated_pose.rotation()).coeffs().transpose() << std::endl;
//    }
//    //std::cout << "Resulting State: " << filter_.state().AllValues.transpose() << std::endl;
//
//
//}

