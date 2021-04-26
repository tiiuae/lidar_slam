//
// Created by sergey on 21.4.2021.
//

#ifndef LIDAR_SLAM_GENERIC_KALMAN_FILTER_H
#define LIDAR_SLAM_GENERIC_KALMAN_FILTER_H

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace lidar_slam
{

/// Generic KalmanState Abstraction
template<std::size_t Size>
class GenericState
{
  protected:
    /// Real container, holding everything (whole Kalman State)
    std::array<float, Size> data_;

  public:
    constexpr static std::size_t StateSize = Size;
    using StateVector = Eigen::Matrix<float, StateSize, 1>;
    using StateMatrix = Eigen::Matrix<float, StateSize, StateSize>;

    /// mapping from @ref data into Eigen::Vector for all State Values
    Eigen::Map<StateVector> AllValues;

    GenericState() : data_ {},
                     AllValues(&data_[0])
    {}

    GenericState(const GenericState<StateSize>& copy)
        : data_ {},
          AllValues(&data_[0])
    {
        std::copy(copy.data_.begin(), copy.data_.end(), data_.begin());
    }

    GenericState<StateSize> operator=(GenericState<StateSize>& copy) = delete;

    virtual StateMatrix Transition(const float time_delta) = 0;
};

class KalmanState : public GenericState<13>
{
  public:
    /// convinience mappings from real array to particular Eigen::Vectors
    Eigen::Map<Eigen::Vector3f> Position;
    Eigen::Map<Eigen::Vector3f> Velocity;
    Eigen::Map<Eigen::Vector3f> AngularVelocity;
    /// We treat Quaternion as non-unit-length quaternion. It's way more easier to integrate such quaternion,
    /// while it's still capable to represent rotation in singularity-free way.
    /// We're intentionally not using Eigen::Quaternionf type, in order to have more control
    /// Quaternion weight is the last component, it is made for convinience of accessing all other fields
    Eigen::Map<Eigen::Vector4f> Quaternion;

    KalmanState() : GenericState<13>(),
                    Position(&data_[0]), Velocity(&data_[3]),
                    AngularVelocity(&data_[6]), Quaternion(&data_[9])
    {

    }


    KalmanState(KalmanState& copy) : GenericState<13>(copy),
                                     Position(&data_[0]), Velocity(&data_[3]),
                                     AngularVelocity(&data_[6]), Quaternion(&data_[9])
    {

    }

    virtual StateMatrix Transition(const float time_delta)
    {
        StateMatrix Transition = StateMatrix::Identity();

        /// same as : Position += Velocity * time_delta;
        Transition(0, 3) = time_delta;
        Transition(1, 4) = time_delta;
        Transition(2, 5) = time_delta;

        /// Quaternion += 0.5F * time_delta * (QuaternionDerivativeXi(Quaternion) * AngularVelocity);
        Transition.block(9, 6, 4, 3) = 0.5F * time_delta * QuaternionDerivativeXi(Quaternion);
        return Transition;
    }

    KalmanState operator=(KalmanState& copy) = delete;

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
};


template<class State>
class GenericKalmanFilter
{
  public:
    constexpr static std::size_t StateSize = State::StateSize;
    using StateMatrix = typename State::StateMatrix;
    using StateVector = typename State::StateVector;

  protected:
    State state_{};
    StateMatrix covariance_{};
    StateMatrix prediction_noise_;
    float timestamp_{};

  public:

    GenericKalmanFilter(const StateVector& initial_state, const StateVector& initial_variance, const StateVector& prediction_noise, float timestamp)
    {
        state_.AllValues = initial_state;
        covariance_ = initial_variance.asDiagonal();
        prediction_noise_ = prediction_noise.asDiagonal();
        timestamp_ = timestamp;
    }

    void Predict(const float new_timestamp)
    {
        const float time_delta = new_timestamp - timestamp_;
        StateMatrix Transition = state_.Transition(time_delta);
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
    void Update(const Eigen::Matrix<float, N, 1>& z,
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

        // Updated (a posteriori) state estimate
        state_.AllValues = state_.AllValues + K * y;

        // Updated (a posteriori) estimate covariance
        covariance_ = (StateMatrix::Identity() - K * H) * covariance_;
    }


};



}  // namespace lidar_slam


#endif  // LIDAR_SLAM_GENERIC_KALMAN_FILTER_H
