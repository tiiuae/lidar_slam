//
// Created by sergey on 19.4.2021.
//
#include <lidar_slam/kalman_filter.h>
#include <iostream>

namespace lidar_slam
{

const static float OriginStateArray[KalmanFilter::StateSize] =
    {0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 1.F};
const KalmanFilter::StateVector KalmanFilter::OriginInitialState = KalmanFilter::StateVector(OriginStateArray);

/// Zero Initial Variance => No doubt in the initial State
const KalmanFilter::StateVector KalmanFilter::NoInitialVariance = KalmanFilter::StateVector::Zero();
/// A little bit uncertain about initial State
const KalmanFilter::StateVector KalmanFilter::SmallInitialVariance = 0.01F * KalmanFilter::StateVector::Ones();

/// Unknown Orientation =>
const static float UnknownOrientationArray[KalmanFilter::StateSize] =
    {0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 1.F, 1.F, 1.F, 1.F};
const KalmanFilter::StateVector KalmanFilter::UnknownOrientationVariance =
    KalmanFilter::StateVector(UnknownOrientationArray);

/// Prediction Process Noise (how well Kalman Prediction explains expected motion)
const KalmanFilter::StateVector KalmanFilter::DefaultPredictionProcessNoise =
    0.001F * KalmanFilter::StateVector::Ones();

KalmanFilter::KalmanFilter(const StateVector& initial_state,
                           const StateVector& initial_variance,
                           const StateVector& prediction_noise,
                           float timestamp)
    : state_{},
      covariance_(initial_variance.asDiagonal()),
      prediction_noise_(prediction_noise.asDiagonal()),
      timestamp_{timestamp}
{
    state_.AllValues = initial_state;

    ObservationModel = Eigen::Matrix<float, 7, StateSize>::Zero();
    ObservationModel(0, 0) = 1;
    ObservationModel(1, 1) = 1;
    ObservationModel(2, 2) = 1;
    ObservationModel(3, 9) = 1;
    ObservationModel(4, 10) = 1;
    ObservationModel(5, 11) = 1;
    ObservationModel(6, 12) = 1;
}

void KalmanFilter::Reset(const StateVector& initial_state,
                         const StateVector& initial_variance,
                         const StateVector& prediction_noise,
                         float timestamp)
{
    state_.AllValues = initial_state;
    covariance_ = initial_variance.asDiagonal();
    prediction_noise_ = prediction_noise.asDiagonal();
    timestamp_ = timestamp;
}

void KalmanFilter::Predict(const float new_timestamp)
{
    const float time_delta = new_timestamp - timestamp_;

    StateMatrix Transition = StateMatrix::Identity();

    /// same as : state_.Position += state_.Velocity * time_delta;
    Transition(0, 3) = time_delta;
    Transition(1, 4) = time_delta;
    Transition(2, 5) = time_delta;

    /// same as : new_state.Pose += 0.5F * time_delta * (QDerivative(state_.Pose) * state_.AngularVelocity);
    Transition.block(9, 6, 4, 3) = 0.5F * time_delta * QuaternionDerivativeXi(state_.Quaternion);

    state_.AllValues = Transition * state_.AllValues;

    // New_Covariance = (Transition * Previous_Covariance  * (Transition)^T ) + Process_Noise
    covariance_ = Transition * covariance_ * Transition.transpose() + prediction_noise_;
}

/// "Generic" function to update Kalman with any kind of measurement
/// @param N (template) - size of observation vector
/// @param z - observation
/// @param H - observation model
/// @param r observation noise
template <unsigned N>
void KalmanFilter::Update(const Eigen::Matrix<float, N, 1>& z,
                          const Eigen::Matrix<float, N, StateSize>& H,
                          Eigen::Matrix<float, N, 1>& r,
                          const float timestamp)
{
    if (timestamp_ < timestamp)
    {
        Predict(timestamp);
    }

    // observation noise matrix
    const Eigen::Matrix<float, N, N> R = r.asDiagonal();

    // Innovation residual
    const Eigen::Matrix<float, N, 1> y = z - H * state_.AllValues;

    // Innovation covariance
    const Eigen::Matrix<float, N, N> S = (H * covariance_ * H.transpose()).eval() + R;

    // Optimal Kalman gain
    const Eigen::Matrix<float, StateSize, N> K = covariance_ * H.transpose() * S.inverse();

    //std::cout << "Optimal Kalman gain K: " << std::endl << K << std::endl;
    //std::cout << "State Update: " << std::endl << (K * y).transpose() << std::endl;

    // Updated (a posteriori) state estimate
    state_.AllValues = state_.AllValues + K * y;

    // Updated (a posteriori) estimate covariance
    covariance_ = (StateMatrix::Identity() - K * H) * covariance_;
}

void KalmanFilter::AbsoluteUpdate(const Eigen::Isometry3f& pose,
                                  const float timestamp,
                                  const float translation_noise,
                                  const float angular_noise)
{
    if (timestamp_ < timestamp)
    {
        Predict(timestamp);
    }

    // Idea here is to prepare unnormalized quaternion, whose weight is the same as weight in the state quaternion
    // It makes both (non-unit) quaternions "compatible" to each other,
    // thus we can claim that we observe only angular components of quaternion
    // Such modification significantly changes property of noise, hopefully not dramatically
    Eigen::Vector3f position = pose.translation();
    Eigen::Quaternionf quaternion(pose.rotation());
    const float qw = state_.Quaternion[3] / quaternion.w();

    Eigen::Matrix<float, 7, 1> Observation{};
    Observation << position, quaternion.x() * qw, quaternion.y() * qw, quaternion.z() * qw, state_.Quaternion[3];

    Eigen::Matrix<float, 7, 1> Noise{};
    Noise << translation_noise, translation_noise, translation_noise, angular_noise, angular_noise, angular_noise, angular_noise;

    Update<7>(Observation, ObservationModel, Noise, timestamp);
}


/// 6Dof Incremental update (Change from last Pose is provided)
void KalmanFilter::IncrementalUpdate(const Eigen::Isometry3f& pose_change,
                                     const float time_delta,
                                     const float translation_noise,
                                     const float angular_noise)
{
        Eigen::Vector3f Velocity = pose_change.translation() / time_delta;

        //Eigen::Quaternionf q(pose_change.rotation());

//        std::cout << "Current Position: [" << state_.Position.transpose() << "] ";
//        std::cout << "Quaternion: [" << state_.Pose.transpose() << "]" << std::endl;
//
//        std::cout << "Update Position: [" << Position.transpose() << "] ";
//        std::cout << "Quaternion: [" << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << "]" << std::endl;

        // Idea here is to prepare unnormalized quaternion, whose weight is the same as weight in the state quaternion
        // It makes both (non-unit) quaternions "compatible" to each other,
        // thus we can claim that we observe only angular components of quaternion
        // Such modification significantly changes property of noise, hopefully not dramatically
//        const float qw = state_.Pose[0] / q.w();
//        Eigen::Vector3f Pose(q.x() * qw, q.y() * qw, q.z() * qw);
//
//        std::cout << "Re-weighted quaternion: [" << q.w() * qw << "," << Pose.transpose() << "] (factor=" << qw << ")"
//                  << std::endl;
//
//        Eigen::Matrix<float, 6, 1> Observation{};
//        Observation << Position, Pose;
//
//        std::cout << "Observation: " << Observation.transpose() << std::endl;
//
//        // std::cout << "ObservationModel: " << std::endl << ObservationModel << std::endl;
//
//        Eigen::Matrix<float, 6, 1> Noise{};
//        Noise << translation_noise, translation_noise, translation_noise, angular_noise, angular_noise, angular_noise;
//
//        Update<6>(Observation, ObservationModel_6DoF, Noise, timestamp);
//
//        std::cout << "Resulting Position: [" << state_.Position.transpose() << "] ";
//        std::cout << "Quaternion: [" << state_.Pose.transpose() << "]" << std::endl;
}

}  // namespace lidar_slam