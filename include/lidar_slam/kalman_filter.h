//
// Created by sergey on 15.4.2021.
//

#ifndef LIDAR_SLAM_KALMAN_FILTER_H
#define LIDAR_SLAM_KALMAN_FILTER_H

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace lidar_slam
{

/// "Generic" 6DoF Kalman Filter
class KalmanFilter
{
  public:
    constexpr static std::size_t StateSize = 13U;
    using StateMatrix = Eigen::Matrix<float, StateSize, StateSize>;
    ///! StateVector holds state parameters in following order:
    /// X, Y, Z  (aka Position)
    /// dX, dY, dZ (aka Velocity)
    /// dAx, dAy, dAz (aka AngularVelocity)
    /// Qx, Qy, Qz, Qw (Quaternion or Pose)
    using StateVector = Eigen::Matrix<float, StateSize, 1>;
    using Covariance = StateMatrix;

    struct State
    {
        /// real container, where everything holds
        std::array<float, StateSize> data;

        /// mapping from @ref data into Eigen::Vector for all State Values
        Eigen::Map<StateVector> AllValues;

        /// convinience mappings from real array to particular Eigen::Vectors
        Eigen::Map<Eigen::Vector3f> Position;
        Eigen::Map<Eigen::Vector3f> Velocity;
        Eigen::Map<Eigen::Vector3f> AngularVelocity;
        /// We treat Quaternion as non-unit-length quaternion. It's way more easier to integrate such quaternion,
        /// while it's still capable to represent rotation in singularity-free way.
        /// We're intentionally not using Eigen::Quaternionf type, in order to have more control
        /// Quaternion weight is the last component, it is made for convinience of accessing all other fields
        Eigen::Map<Eigen::Vector4f> Quaternion;

        State() : data {},
              AllValues(&data[0]), Position(&data[0]), Velocity(&data[3]),
              AngularVelocity(&data[6]), Quaternion(&data[9]) {}

        State(State& copy)
            : data {},
              AllValues(&data[0]), Position(&data[0]), Velocity(&data[3]),
              AngularVelocity(&data[6]), Quaternion(&data[9])
        {
            std::copy(copy.data.begin(), copy.data.end(), data.begin());
        }

        State operator=(State& copy) = delete;
    };

    KalmanFilter(const StateVector& initial_state, const StateVector& initial_variance, const StateVector& prediction_noise, float timestamp);

    void Reset(const StateVector& initial_state, const StateVector& initial_variance, const StateVector& prediction_noise, float timestamp);

    void Predict(const float new_timestamp);

    /// "Generic" function to update Kalman with any kind of measurement
    /// @param N (template) - size of observation vector
    /// @param z - observation
    /// @param H - observation model
    /// @param r observation noise
    template <unsigned N>
    void Update(const Eigen::Matrix<float, N, 1>& z,
                const Eigen::Matrix<float, N, StateSize>& H,
                Eigen::Matrix<float, N, 1>& r,
                const float timestamp);


    /// Kalman update, using absolute (world-coordinate) 6Dof Pose
    void AbsoluteUpdate(const Eigen::Isometry3f& pose,
                const float timestamp,
                const float translation_noise = 0.01,
                const float angular_noise = 0.001);

    /// Kalman update, using incremental
    void IncrementalUpdate(const Eigen::Isometry3f& current2new_transform,
                        const float timestamp,
                        const float translation_noise = 0.01,
                        const float angular_noise = 0.001)
    {
        if(timestamp > timestamp_)
        {
            const Eigen::Isometry3f current = Pose();
            const Eigen::Isometry3f new_pose = current2new_transform * current;
            AbsoluteUpdate(new_pose, timestamp, translation_noise, angular_noise);
        }
    }

    Eigen::Isometry3f Pose()
    {
        Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
        Eigen::Quaternionf q(state_.Quaternion[3], state_.Quaternion[0], state_.Quaternion[1], state_.Quaternion[2]);
        q.normalize();
        pose.rotate(q);
        pose.pretranslate(state_.Position);
        return pose;
    }

    float timestamp()
    {
        return timestamp_;
    }

    State state()
    {
        return state_;
    }

    StateMatrix covariance()
    {
        return covariance_;
    }

    /// Origin Pose - everything is zero, except for quaternion weight, which is one
    const static StateVector OriginInitialState;
    /// Zero Initial Variance => No doubt in the initial State
    const static KalmanFilter::StateVector NoInitialVariance;
    /// A little bit uncertain about initial State
    const static KalmanFilter::StateVector SmallInitialVariance;
    /// Unknown Orientation variance
    const static KalmanFilter::StateVector UnknownOrientationVariance;
    /// Prediction Process Noise (how well Kalman Prediction explains expected motion)
    const static KalmanFilter::StateVector DefaultPredictionProcessNoise;


  private:

    /// Quaternion Derivative Multiplication Matrix, where Q is in form [Qx, Qy, Qz, Qw]
    /// dQ/dt = 1/2 * Xi(Q) * W (AngularVelocity, which is pure quaternion) * dt
    static Eigen::Matrix<float, 4, 3> QuaternionDerivativeXi(const Eigen::Vector4f& Q)
    {
        // https://math.stackexchange.com/questions/189185/quaternion-differentiation
        Eigen::Matrix<float, 4, 3> Xi{};
        Xi.row(0) << Q[3], Q[2], -Q[1];
        Xi.row(1) << -Q[2], Q[3], Q[0];
        Xi.row(2) << Q[1], -Q[0], Q[3];
        Xi.row(3) << -Q[0], -Q[1], -Q[2];
        return Xi;
    }

//    static Eigen::Matrix<float, 4, 4> QuaternionDerivativeOmega(const Eigen::Vector3f& w)
//    {
//        // https://math.stackexchange.com/questions/189185/quaternion-differentiation
//        Eigen::Matrix<float, 4, 3> Omega{};
//        Omega.row(0) << 0, -w[0], -w[1], -w[2];
//        Omega.row(1) << w[0], 0, w[1], -w[2];
//        Omega.row(2) << w[1], -w[2], 0, w[0];
//        Omega.row(3) << w[2], w[1], -w[0], 0;
//        return Omega;
//    }


    State state_{};
    StateMatrix covariance_{};
    StateMatrix prediction_noise_;
    float timestamp_{};

    Eigen::Matrix<float, 7, StateSize> AbsoluteObservationModel;
};




}  // namespace lidar_slam

#endif  // LIDAR_SLAM_KALMAN_FILTER_H
