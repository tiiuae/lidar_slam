#include <lidar_slam/smart_voxeling.h>

using namespace lidar_slam;

#ifdef USE_OWN_FEATURES
public:
    // List of "radius" (in meters) for feature calculation
    constexpr static std::array<float, 4> FeatureScales{0.1F, 0.5F, 1.F, 2.F};

  private:
    void static FeatureDetector(const PointCloud& cloud);
    static const std::size_t BufferSize = 5U;
    std::array<CloudMsg::SharedPtr, BufferSize> buffer_;
    std::size_t buffer_head_, buffer_tail_;
#endif
