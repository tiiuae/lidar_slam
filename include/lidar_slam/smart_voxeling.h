//
// Created by sergey on 23.3.2021.
//

#ifndef FOG_SW_CLOUD_ALIGNER_H
#define FOG_SW_CLOUD_ALIGNER_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <unordered_map>
#include <cstdint>

namespace lidar_slam
{

/// Fast and Smart Voxeling, computing mean (i.e. centroid) and covariance of each voxel in one pass
template<typename PointType>
class SmartVoxeling
{
  public:
    using PointCloud = pcl::PointCloud<PointType>;
    using PointCloudPtr = PointCloud::Ptr;
    using IndexType = std::size_t;
    struct Voxel3D {

        Eigen::Vector3f mean {};

        /// "squared distances" buffer, required for "running" variance estimate
        Eigen::Vector3f M2 {};

        /// running variance
        Eigen::Vector3f variance {};

        // number of points in the voxel
        std::size_t size = 0U;

        Point distinctive {};
        float largest_distance {0.F};

        /// Adds points into the voxel and calculates centroid
        /// May not add any points after normalization
        void AddPoint(const Eigen::Vector3f& p)
        {
            if(size > 0U)
            {
                Eigen::Vector3f diff = p - mean;
                size ++;

            }
            else
            {
                mean = p;
                M2 = {};
                variance = {};
                size++;
            }



//            if(!normalized)
//            {
//                x += p.x;
//                y += p.y;
//                z += p.z;
//                size++;
//            }
        }
//
//        /// once all points in the cloud are processed you need to normalize each voxel in order to caclulate centroid
//        void Normalize()
//        {
//            if(!normalized && size > 2U)
//            {
//                x /= float(size);
//                y /= float(size);
//                z /= float(size);
//                normalized = true;
//            }
//        }
//
//        /// Once all the centrids are calculated one may compare what's the
//        void AddPointAgain(const Point& p)
//        {
//            if(normalized)
//            {
//                const float distance = (p.x-x)*(p.x-x)+(p.y-y)*(p.y-y)+(p.z-z)*(p.z-z);
//                if(distance > largest_distance)
//                {
//                    distinctive = p;
//                    largest_distance = distance;
//                }
//
//            }
//
//        }

    };

    SmartVoxeling(const PointCloudPtr& cloud, const float voxel_size = 1.F) :
        voxel_size_{voxel_size}, voxels_{}, voxel_cloud_{}
    {
        // create voxels
        for(std::size_t i = 0; i < cloud->size(); i++)
        {
            Point p = cloud->at(i);
            voxels_(p).addPoint(p);
        }

//        // normalize
//        for(auto voxel : voxels_)
//        {
//            voxel.second.normalize();
//            Point p(voxel.second.x, voxel.second.y, voxel.second.z);
//            voxel_cloud_.push_back(p);
//        }

    }


    /// Returns all voxel centroids as separate point cloud
    PointCloudPtr VoxelCloud()
    {
        return voxel_cloud_;
    }

    Voxel& operator[] (const IndexType voxelIndex)
    {
        return voxels_[voxelID];
    }

    Voxel operator[] (const IndexType voxelIndex) const
    {
        return voxels_.at(voxelID);
    }

    Voxel& operator() (const PointType& point)
    {
        return voxels_[Index(point)];
    }

    Voxel operator() (const PointType& point) const
    {
        return voxels_.at(Index(point));
    }

    /// Indexing (aka Hashing) is required to map 3 integer voxel coordinates into one "hash"
    /// It goes in "spiral" manner, i.e. voxel (0,0,0) has index=0
    /// Nearby voxels with radius=1 and coordinates (-1..1, -1..1, -1..1) have indexes 1..26
    /// Next radius has indexes within range 27..4^3
    /// Voxels of radius i have indexes in the range (w(i)^3 - w(i-1)^3)..k^3, where w(i) = 2*i+1
    IndexType Index(const PointType& point) const
    {
        // integer coordinates of the voxel, in range (-radius ... + radius)
        const std::int32_t ix = std::int32_t(roundf(point.x / voxel_size_));
        const std::int32_t iy = std::int32_t(roundf(point.y / voxel_size_));
        const std::int32_t iz = std::int32_t(roundf(point.z / voxel_size_));

        // radius of the layer
        const std::int32_t radius = std::max(std::abs(ix), std::max(std::abs(iy), std::abs(iz)));

        if(radius == 0)
        {
            return 0U;
        }

        // absolute coordinates of the voxel, going in range (0...length-1)
        const std::uint32_t ax = std::uint32_t(ix + radius);
        const std::uint32_t ay = std::uint32_t(iy + radius);
        const std::uint32_t az = std::uint32_t(iz + radius);

        const std::uint32_t prev_length = (std::uint32_t(radius) - 1U) * 2U + 1U;
        const std::uint32_t length = std::uint32_t(radius) * 2U + 1U; // number of voxels in one edge
        const std::uint32_t side = length * length; // number of voxels in 2D side

        const std::uint64_t end_index = std::uint64_t(length*length*length);
        const std::uint64_t start_index = end_index - std::uint64_t(prev_length*prev_length*prev_length);

        // now we have situation where at least one coordinate must be 0 or length-1
        // this allows to simplify index hash computation
        // we don't really need "reversble" hash, it's just need to be unique
        // so we treat each side of the "cube" separately
        if(ax == 0U)
        {
            return start_index + std::uint64_t(ay * length) + std::uint64_t(az);
        }

        if(ax == length - 1U)
        {
            return start_index + side + std::uint64_t(ay * length) + std::uint64_t(az);
        }

        if(ay == 0)
        {
            return start_index + 2U*side + std::uint64_t(az * (length-2U)) + std::uint64_t(ax-1U);
        }

        if(ay == length - 1U)
        {
            return start_index + 2U*side + length*(length-2U) + std::uint64_t(az * (length-2U) + (ax-1U));
        }

        if(az == 0)
        {
            return start_index + 2U*side + 2U*length*(length-2U) + std::uint64_t((ax-1U) * (length-2U) + (ay-1U));
        }

        return start_index + 2U*side + 2U*length*(length-2U) + (length-2U)*(length-2U) + std::uint64_t((ax-1U) * (length-2U) + (ay-1U));
    }



    ///
    PointCloudPtr DistinctivePoints()
    {

        return voxel_cloud_;
    }



  protected:
    float voxel_size_;
    std::unordered_map<IndexType, Voxel3D> voxels_;
    PointCloudPtr voxel_cloud_;
    PointCloudPtr distinctive_points_;
};

//
//template<typename PointType, unsigned levels=2>
//class HierarchicalVoxeling
//{
//  public:
//    using PointCloud = pcl::PointCloud<PointType>;
//    using PointCloudPtr = PointCloud::Ptr;
//    using IndexType = std::size_t;
//
//
//    ///
//    HierarchicalVoxeling(const PointCloudPtr& cloud, const float voxel_radius = 1.F)
//    {
//
//    }
//

//
//
//  protected:
//
//
//    PointCloudPtr voxel_cloud_;
//
//};

//template<typename PointType, unsigned levels=2>
//class CloudAligner
//{
//
//};

} //namespace lidar_slam
//
//#ifdef USE_OWN_FEATURES
//void LidarSlam::FeatureDetector(const PointCloud& cloud)
//{
//    if (cloud.width == 1 || cloud.height == 1)
//    {
//        return;
//    }
//
//    struct PlanarFeature
//    {
//        float x, y, z;        // centroid coordinates
//        float a, b, c;        // estimated "plane" coordinates: z = ax + by + c
//        Eigen::Vector3f ZXt;  // Z X^T
//        Eigen::Matrix3f XXt;  //
//        std::size_t amount;
//    };
//
//    std::size_t width = cloud.width;
//    std::size_t height = cloud.height;
//    std::vector<std::vector<PlanarFeature>> features{};
//
//    for (std::size_t s = 0; s < FeatureScales.size(); s++)
//    {
//        const float scale = FeatureScales[s];
//        const std::size_t new_width = (width - 1U) / 2U;
//        const std::size_t new_height = (height - 1U) / 2U;
//        std::vector<PlanarFeature> planes0(new_width * new_height);
//        features.push_back(planes0);
//
//        // left-to-right pass
//        for (std::size_t new_y = 0U; new_y < new_height; new_y++)
//        {
//            for (std::size_t ox = 0U; ox < width - 1U; ox++)
//            {
//            }
//        }
//    }
//}
//#endif

#endif  // FOG_SW_CLOUD_ALIGNER_H
