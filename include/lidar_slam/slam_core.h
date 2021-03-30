//
// Created by sergey on 23.3.2021.
//

#ifndef FOG_SW_SLAM_CORE_H
#define FOG_SW_SLAM_CORE_H

#include <g2o/types/slam3d/edge_se3.h>

namespace lidar_slam
{

/// SLAM Backend part
class SlamCore
{
  public:
    SlamCore() = default;

    const std::vector<g2o::VertexSE3*>&  poses();
    const std::vector<g2o::VertexSE3*>&  points();
    const std::vector<g2o::EdgeSE3*>&  edges();


};


} //namespace lidar_slam
#endif  // FOG_SW_SLAM_CORE_H
