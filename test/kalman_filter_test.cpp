#include <gtest/gtest.h>
#include <lidar_slam/kalman_filter.h>

using namespace lidar_slam;
using namespace Eigen;

class KalmanFilterTest : public ::testing::Test
{
  public:
    KalmanFilterTest() :

        filter_(OriginInitialState, SmallInitialVariance, DefaultPredictionProcessNoise, 1.F)
    {

    }

  protected:

    KalmanFilter filter_;

};


/// Basic Prediction test
TEST_F(KalmanFilterTest, BasicPredictTest)
{
    EXPECT_EQ(0.F, filter_.state().Position.norm());
    EXPECT_EQ(1.F, filter_.state().Pose.norm());
    EXPECT_EQ(1.F, filter_.state().Pose[0]);
    EXPECT_EQ(0.F, filter_.state().Velocity.norm());
    EXPECT_EQ(0.F, filter_.state().AngularVelocity.norm());

    filter_.Predict(2.F);
    EXPECT_EQ(0.F, filter_.state().Position.norm());
    EXPECT_EQ(1.F, filter_.state().Pose.norm());
    EXPECT_EQ(1.F, filter_.state().Pose[0]);
    EXPECT_EQ(0.F, filter_.state().Velocity.norm());
    EXPECT_EQ(0.F, filter_.state().AngularVelocity.norm());
}

/// Basic Update test
TEST_F(KalmanFilterTest, BasicUpdateTest)
{
    Isometry3f iso = Isometry3f::Identity();
    iso.pretranslate(Vector3f(1, 0 ,0));
    //std::cout << "Initial Covariance: " << std::endl <<  filter_.covariance() << std::endl;
    filter_.Update(iso, 2.F);
    //std::cout << "A-posteriori Covariance: " << std::endl <<  filter_.covariance() << std::endl;

    const float x = filter_.state().Position[0];
    const float dx = filter_.state().Velocity[0];

    EXPECT_TRUE(1.F > x && x > 0.F ) << "x=" << x;
    EXPECT_EQ(0.F, filter_.state().Position[1]);
    EXPECT_EQ(0.F, filter_.state().Position[2]);
    EXPECT_TRUE(1.F > dx && dx > 0.F ) << "dx=" << dx;
    EXPECT_EQ(0.F, filter_.state().Velocity[1]);
    EXPECT_EQ(0.F, filter_.state().Velocity[2]);
    EXPECT_EQ(1.F, filter_.state().Pose.norm());
    EXPECT_EQ(1.F, filter_.state().Pose[0]);
    EXPECT_EQ(0.F, filter_.state().AngularVelocity.norm());
}


/// Basic Rotation test
TEST_F(KalmanFilterTest, BasicRotationTest)
{
    Isometry3f iso = Isometry3f::Identity();
    iso.pretranslate(Vector3f(1, 0 ,0));
    Eigen::AngleAxisf x_rot(1.F, Eigen::Vector3f::UnitX());
    iso.prerotate(x_rot);

    std::cout << "Initial Covariance: " << std::endl <<  filter_.covariance() << std::endl;
    filter_.Update(iso, 2.F);
    std::cout << "A-posteriori Covariance: " << std::endl <<  filter_.covariance() << std::endl;

    const float x = filter_.state().Position[0];
    const float dx = filter_.state().Velocity[0];
    const float ax = filter_.state().Pose[1];
    const float wx = filter_.state().AngularVelocity[0];

    EXPECT_TRUE(1.F > x && x > 0.F ) << "x=" << x;
    EXPECT_EQ(0.F, filter_.state().Position[1]);
    EXPECT_EQ(0.F, filter_.state().Position[2]);
    EXPECT_TRUE(1.F > dx && dx > 0.F ) << "dx=" << dx;
    EXPECT_EQ(0.F, filter_.state().Velocity[1]);
    EXPECT_EQ(0.F, filter_.state().Velocity[2]);
    EXPECT_LT(1.F, filter_.state().Pose.norm());
    EXPECT_EQ(1.F, filter_.state().Pose[0]);
    EXPECT_TRUE(1.F > ax && ax > 0.F) << "ax=" << ax;
    EXPECT_TRUE(1.F > wx && wx > 0.F) << "wx=" << wx;

    EXPECT_EQ(0.F, filter_.state().Pose[2]);
    EXPECT_EQ(0.F, filter_.state().Pose[3]);
    EXPECT_EQ(0.F, filter_.state().AngularVelocity[1]);
    EXPECT_EQ(0.F, filter_.state().AngularVelocity[2]);

    filter_.Predict(3.F);

    const float x2 = filter_.state().Position[0];
    const float dx2 = filter_.state().Velocity[0];
    const float ax2 = filter_.state().Pose[1];
    const float wx2 = filter_.state().AngularVelocity[0];

    // Since velocities are positive, we shall witness increase in x and ax
    EXPECT_TRUE(1.F > x2 && x2 > x ) << "x=" << x << ", x2=" << x2;
    EXPECT_TRUE( dx2 ==  dx ) << "dx=" << dx << ", dx2=" << dx2;
    EXPECT_TRUE(1.F > ax2 && ax2 > ax ) << "ax=" << ax << ", ax2=" << x2;
    EXPECT_TRUE( wx2 == wx ) << "wx=" << wx << ", wx2=" << x2;

}