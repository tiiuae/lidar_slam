//
// Created by sergey on 15.4.2021.
//

#ifndef LIDAR_SLAM_KALMAN_FILTER_H
#define LIDAR_SLAM_KALMAN_FILTER_H

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace lidar_slam
{

class KalmanFilter
{
  public:
    constexpr static std::size_t StateSize = 13U;
    using StateMatrix = Eigen::Matrix<float, StateSize, StateSize>;
    using StateVector = Eigen::Matrix<float, StateSize, 1>;
    using Covariance = StateMatrix;

    struct State
    {
        std::array<float, StateSize> data;

        Eigen::Map<StateVector> Value;
        Eigen::Map<Eigen::Vector3f> Position;
        Eigen::Map<Eigen::Vector3f> Velocity;
        /// We treat Pose as non-unit-length quaternion. It's way more easier to integrate such quaternion,
        /// while it's still capable to represent rotation in singularity-free way.
        /// We're intenionally not using Eigen::Quaternionf type, in order to to have more control
        Eigen::Map<Eigen::Vector4f> Pose;
        Eigen::Map<Eigen::Vector3f> AngularVelocity;

        State() : Value(&data[0]), Position(&data[0]), Velocity(&data[3]), Pose(&data[6]), AngularVelocity(&data[10]) {}

        State(State& copy)
            : Value(&data[0]), Position(&data[0]), Velocity(&data[3]), Pose(&data[6]), AngularVelocity(&data[10])
        {
            std::copy(copy.data.begin(), copy.data.end(), data.begin());
        }

        State operator=(State& copy) = delete;
    };

    KalmanFilter(const StateVector& initial_state, const StateVector& initial_variance, const StateVector& prediction_noise, float timestamp)
        : state_{},
          covariance_(initial_variance.asDiagonal()),
          prediction_noise_(prediction_noise.asDiagonal()),
          timestamp_{timestamp}
    {
        state_.Value = initial_state;

    };

    void Reset(const StateVector& initial_state, const StateVector& initial_variance, const StateVector& prediction_noise, float timestamp)
    {
        state_.Value = initial_state;
        covariance_ = initial_variance.asDiagonal();
        prediction_noise_ = prediction_noise.asDiagonal();
        timestamp_ = timestamp;
    }

    void Predict(const float new_timestamp)
    {
        const float time_delta = new_timestamp - timestamp_;

        StateMatrix Transition = StateMatrix::Identity();

        /// same as : state_.Position += state_.Velocity * time_delta;
        Transition(0, 3) = time_delta;
        Transition(1, 4) = time_delta;
        Transition(2, 5) = time_delta;

        /// same as : new_state.Pose += 0.5F * time_delta * (QDerivative(state_.Pose) * state_.AngularVelocity);
        Transition.block(6, 10, 4, 3) = 0.5F * time_delta * QuaternionDerivative(state_.Pose);

        state_.Value = Transition * state_.Value;

        // New_Covariance = (Transition * Previous_Covariance  * (Transition)^T ) + Process_Noise
        covariance_ = Transition * covariance_ * Transition.transpose() + prediction_noise_;
    }

    /// "Generic" function to update Kalman with any kind of measurement
    /// @param N (template) - size of observation vector
    /// @param z - observation
    /// @param H - observation model
    /// @param r observation noise
    template <unsigned N>
    void Update(const Eigen::Matrix<float, N, 1>& z,
                const Eigen::Matrix<float, N, StateSize>& H,
                Eigen::Matrix<float, N, 1>& r,
                const float timestamp)
    {
        if (timestamp_ != timestamp)
        {
            Predict(timestamp);
        }

        const Eigen::Matrix<float, N, N> R = r.asDiagonal();

        // Innovation residual
        const Eigen::Matrix<float, N, 1> y = z - H * state_.Value;

        // Innovation covariance
        const Eigen::Matrix<float, N, N> S = (H * covariance_ * H.transpose()).eval() + R;

        // Optimal Kalman gain
        const Eigen::Matrix<float, StateSize, N> K = covariance_ * H.transpose() * S.inverse();

        // Updated (a posteriori) state estimate
        state_.Value = state_.Value + K * y;

        // Updated (a posteriori) estimate covariance
        covariance_ = (StateMatrix::Identity() - K * H) * covariance_;
    }

    /// Makes  update, meaning that estmate is given not
    void Update(const Eigen::Isometry3f& pose,
                const float timestamp,
                const float translation_noise = 0.01,
                const float angular_noise = 0.001)
    {
        Eigen::Vector3f Position = pose.translation();
        Eigen::Quaternionf q(pose.rotation());

        std::cout << "Current Position: [" << state_.Position.transpose() << "] ";
        std::cout << "Quaternion: [" <<  state_.Pose.transpose() << "]" << std::endl;

        std::cout << "Update Position: [" << Position.transpose() << "] ";
        std::cout << "Quaternion: [" << q.w() << "," <<  q.x() << "," <<  q.y()<< "," <<  q.z() << "]" << std::endl;

        // Idea here is to prepare unnormalized quaternion, whose weight is the same as weight in the state quaternion
        // It makes both (non-unit) quaternions "compatible" to each other,
        // thus we can claim that we observe only angular components of quaternion
        // Such modification significantly changes property of noise, hopefully not dramatically
        const float qw = state_.Pose[0] / q.w();
        Eigen::Vector3f Pose(q.x() * qw, q.y() * qw, q.z() * qw);

        std::cout << "Re-weighted quaternion: [" << q.w()*qw << "," << Pose.transpose() << "] (factor="<< qw << ")" << std::endl;

        Eigen::Matrix<float, 6, 1> Observation{};
        Observation << Position, Pose;

        std::cout << "Observation: " << Observation .transpose() << std::endl;

        Eigen::Matrix<float, 6, StateSize> ObservationModel = Eigen::Matrix<float, 6, StateSize>::Zero();
        ObservationModel(0, 0) = 1;
        ObservationModel(1, 1) = 1;
        ObservationModel(2, 2) = 1;
        ObservationModel(3, 7) = 1;
        ObservationModel(4, 8) = 1;
        ObservationModel(5, 9) = 1;

        //std::cout << "ObservationModel: " << std::endl << ObservationModel << std::endl;

        Eigen::Matrix<float, 6, 1> Noise{};
        Noise << translation_noise, translation_noise, translation_noise, angular_noise, angular_noise, angular_noise;

        Update<6>(Observation, ObservationModel, Noise, timestamp);

        std::cout << "Resulting Position: [" << state_.Position.transpose() << "] ";
        std::cout << "Quaternion: [" <<  state_.Pose.transpose() << "]" << std::endl;
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

  private:
    /// Quaternion Derivative
    static Eigen::Matrix<float, 4, 3> QuaternionDerivative(const Eigen::Vector4f& q)
    {
        // https://math.stackexchange.com/questions/189185/quaternion-differentiation
        Eigen::Matrix<float, 4, 3> omega{};
        omega.row(0) << -q[1], -q[2], -q[3];
        omega.row(1) << q[0], q[3], -q[2];
        omega.row(2) << -q[3], q[0], q[1];
        omega.row(3) << q[2], -q[1], q[0];
        return omega;
    }

    State state_{};
    StateMatrix covariance_{};
    StateMatrix prediction_noise_;
    float timestamp_{};
};

/// Origin Pose - everything is zero, except for quaternion weight, which is one
const float OriginStateArray[KalmanFilter::StateSize] = {0.F,0.F,0.F,0.F,0.F,0.F,1.F,0.F,0.F,0.F,0.F,0.F,0.F};
const KalmanFilter::StateVector OriginInitialState = KalmanFilter::StateVector(OriginStateArray);

/// Zero Initial Variance => No doubt in the initial State
const KalmanFilter::StateVector NoInitialVariance = KalmanFilter::StateVector::Zero();
/// A little bit uncertain about initial State
const KalmanFilter::StateVector SmallInitialVariance = 0.001F * KalmanFilter::StateVector::Ones();

/// Unknown Orientation =>
const float UnknownOrientationArray[KalmanFilter::StateSize] = {0.F,0.F,0.F,0.F,0.F,0.F,1.F,1.F,1.F,1.F,0.F,0.F,0.F};
const KalmanFilter::StateVector UnknownOrientationInitialVariance = KalmanFilter::StateVector(UnknownOrientationArray);

/// Prediction Process Noise (how well Kalman Prediction explains expected motion)
const KalmanFilter::StateVector DefaultPredictionProcessNoise = 0.001F * KalmanFilter::StateVector::Ones();


}  // namespace lidar_slam

#endif  // LIDAR_SLAM_KALMAN_FILTER_H
