//
// Created by sergey on 21.4.2021.
//

#ifndef LIDAR_SLAM_IMU_DEAD_RECKONING_H
#define LIDAR_SLAM_IMU_DEAD_RECKONING_H

#include "generic_kalman_filter.h"
namespace lidar_slam
{

///! ImuKalmanState holds state parameters in following order:
/// X, Y, Z  (aka Position)
/// dX, dY, dZ (aka Velocity)
/// aX, aY, aZ (aka Acceleration)
/// Qx, Qy, Qz, Qw (Quaternion or Pose)
/// dAx, dAy, dAz (aka AngularVelocity)

/// GXo, GYo, GZo (GyroBias)
/// AXo, AYo, AZo (AccelBias)
class ImuKalmanState : public GenericState<16>
{
  public:
    constexpr static std::size_t StateSize = GenericState<16>::StateSize;

    /// convinience mappings from real array to particular Eigen::Vectors
    Eigen::Map<Eigen::Vector3f> Position;
    Eigen::Map<Eigen::Vector3f> Velocity;
    Eigen::Map<Eigen::Vector3f> Acceleration;

    /// We treat Quaternion as non-unit-length quaternion. It's way more easier to integrate such quaternion,
    /// while it's still capable to represent rotation in singularity-free way.
    /// We're intentionally not using Eigen::Quaternionf type, in order to have more control
    /// Quaternion weight is the last component, it is made for convinience of accessing all other fields
    Eigen::Map<Eigen::Vector4f> Quaternion;
    Eigen::Map<Eigen::Vector3f> AngularVelocity;
    //Eigen::Map<Eigen::Vector3f> AccelBias;
    //Eigen::Map<Eigen::Vector3f> GyroBias;



    ImuKalmanState() : GenericState<StateSize>(),
                    Position(&data_[0]),
                    Velocity(&data_[3]),
                    Acceleration(&data_[6]),
                    Quaternion(&data_[9]),
                    AngularVelocity(&data_[13])
                    //AccelBias(&data_[16]),
                    //GyroBias(&data_[19])
    {

    }


    ImuKalmanState(ImuKalmanState& copy) : GenericState<StateSize>(copy),
                                     Position(&data_[0]), Velocity(&data_[3]),
                                     AngularVelocity(&data_[6]), Quaternion(&data_[9]),
                                     GyroBias(&data_[13]), AccelBias(&data_[16])
    {

    }

    virtual StateMatrix Transition(const float time_delta)
    {
        // New State = Old State
        StateMatrix Transition = StateMatrix::Identity();

        // New.Position += Old.Velocity * time_delta/2;
        Transition.block(0,3,3,3) = Eigen::Matrix3f::Identity() * time_delta/2.F;

        // New.Position += Old.Acceleration * time_delta^2/4;
        Transition.block(0,6,3,3) = Eigen::Matrix3f::Identity() * (time_delta*time_delta/4.F);

        /// New.Velocity += Old.Acceleration * time_delta/2;
        Transition.block(3,6,3,3) = Eigen::Matrix3f::Identity() * time_delta/2.F;

        /// New.Quaternion += (QuaternionDerivativeXi(Old.Quaternion) * Old.AngularVelocity) * time_delta/2;
        Transition.block(9, 13, 4, 3) = QuaternionDerivativeXi(Quaternion) * time_delta/2.F;

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


/// For IMU Dead Reckoning we extend generic KF
/// However, we do not expose KF interface
class ImuDeadReckoning : private GenericKalmanFilter<ImuKalmanState>
{
  public:
    //constexpr static std::size_t StateSize = 19U;
    //using StateMatrix = Eigen::Matrix<float, StateSize, StateSize>;
    //using StateVector = Eigen::Matrix<float, StateSize, 1>;


    ImuKalmanFilter(const StateVector& initial_state, const StateVector& initial_variance, const StateVector& prediction_noise, float timestamp)
        : state_{},
          covariance_(initial_variance.asDiagonal()),
          prediction_noise_(prediction_noise.asDiagonal()),
          timestamp_{timestamp}
    {
        state_.AllValues = initial_state;

        ObservationModel = Eigen::Matrix<float, 19, StateSize>::Zero();
        // Acceleration - comes from accel
        ObservationModel(0, 6) = 1;
        ObservationModel(1, 7) = 1;
        ObservationModel(2, 8) = 1;
        // Quaternion - only partialloy observable
        ObservationModel(3, 9) = 1;
        ObservationModel(4, 10) = 1;
        ObservationModel(5, 11) = 1;
        ObservationModel(6, 12) = 1;
        // AngularVelocity - comes from gyro
        ObservationModel(7, 13) = 1;
        ObservationModel(8, 14) = 1;
        ObservationModel(9, 15) = 1;

    }





    void Reset(const StateVector& initial_state, const StateVector& initial_variance, const StateVector& prediction_noise, float timestamp)
    {

        state_.AllValues = initial_state;
        covariance_ = initial_variance.asDiagonal();
        prediction_noise_ = prediction_noise.asDiagonal();
        timestamp_ = timestamp;

    }

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

    struct ImuData
    {
        Eigen::Vector3f gyro; //[radians/sec]
        Eigen::Vector3f accel; // N
        Eigen::Vector3f mag; // microtesla
        float timestamp;
    };


    void Integrate(const ImuData &imu)
    {
        const bool looks_stationary = (imu.gyro.norm() < 0.1F) && (std::abs(imu.accel.norm() - 9.8F) < 0.1F);

        stationary_probability_ *= (looks_stationary ? 1.2F : 0.9F);
        stationary_probability_ = std::min(1.F, std::max(0.F, stationary_probability_));

        Eigen::Vector3f up (0,0,1);
        //Eigen::Vector3f north (1,0,0);
        if(stationary_probability_ > 0.99F)
        {
            state_.AngularVelocity = Eigen::Vector3f::Zero();
            state_.Velocity = Eigen::Vector3f::Zero();
        }
        else
        {
            const float time_delta = imu.timestamp - timestamp_;
            state_.AngularVelocity = imu.gyro * time_delta;
            auto CompensatedAccel = imu.accel - Rotation() * gravity_;
            state_.Velocity += CompensatedAccel * time_delta;
        }

        Predict(imu.timestamp);
    }

    void Integrate2(const ImuData &imu)
    {
        const bool looks_stationary = (imu.gyro.norm() < 0.1F) && (std::abs(imu.accel.norm() - 9.8F) < 0.1F);
        Predict(imu.timestamp);
        stationary_probability_ *= (looks_stationary ? 1.2F : 0.9F);
        stationary_probability_ = std::min(1.F, std::max(0.F, stationary_probability_));

        Eigen::Vector3f up (0,0,1);
        //Eigen::Vector3f north (1,0,0);


        if(stationary_probability_ > 0.99F)
        {

            state_.AngularVelocity = Eigen::Vector3f::Zero();
            state_.Velocity = Eigen::Vector3f::Zero();
        }
        else
        {
            const float time_delta = imu.timestamp - timestamp_;
            state_.AngularVelocity = imu.gyro * time_delta;
            auto CompensatedAccel = imu.accel - Rotation() * gravity_;
            state_.Velocity += CompensatedAccel * time_delta;
        }


    }


  private:

    const Eigen::Vector3f gravity_ {0,0,-9.8};
    Eigen::Matrix<float, 19, StateSize> ObservationModel;
    float stationary_probability_ = 0.5F;
};

}  // namespace lidar_slam


#endif  // LIDAR_SLAM_IMU_DEAD_RECKONING_H
