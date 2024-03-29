cmake_minimum_required(VERSION 3.5)
project(lidar_slam)

# Default to C99
if (NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
endif ()

# Default to C++14
if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
endif ()

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    #add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -pthread -O3")

set(OpenGL_GL_PREFERENCE "LEGACY")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(g2o REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(OpenGL COMPONENTS OpenGL)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW glfw3)
find_package(rosbag2 REQUIRED)
find_package(rosbag2_cpp REQUIRED)


include_directories(SYSTEM ${PCL_INCLUDE_DIRS})
include_directories(SYSTEM "/usr/include/suitesparse") # dirty hack, needs to be fixed by proper find_package
include_directories(include)

add_definitions(${PCL_DEFINITIONS})

option(USE_SLAM_LOGGING False)
if(${USE_SLAM_LOGGING})
    add_compile_definitions(USE_SLAM_LOGGING)
endif()


add_library(lidar_slam src/lidar_slam.cpp)
target_link_libraries(lidar_slam g2o::core g2o::types_icp g2o::types_sba g2o::solver_cholmod ${PCL_LIBRARIES} OpenMP::OpenMP_CXX)

#ament_export_dependencies(ament_cmake rclcpp g2o::core g2o::types_icp g2o::types_sba g2o::solver_cholmod ${PCL_LIBRARIES} OpenMP::OpenMP_CXX)
#ament_export_dependencies(ament_cmake rclcpp Eigen3 px4_msgs geometry_msgs sensor_msgs)
ament_export_dependencies(PCL Eigen3)

#ament_export_targets(lidar_slam)
#ament_export_interfaces(export_lidar_slam HAS_LIBRARY_TARGET)
ament_export_include_directories(include ${PCL_INCLUDE_DIRS})
#ament_export_libraries(lidar_slam)

install(DIRECTORY include/lidar_slam
        DESTINATION include)

install(TARGETS lidar_slam
        EXPORT export_lidar_slam
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )

add_executable(lidar_slam_node src/lidar_slam_node.cpp )
ament_target_dependencies(lidar_slam_node rclcpp sensor_msgs tf2 tf2_ros nav_msgs pcl_conversions)
target_link_libraries(lidar_slam_node lidar_slam)
install(TARGETS lidar_slam_node DESTINATION lib/${PROJECT_NAME} )

#add_executable(handeye_calibration src/handeye_calibration.cpp)
#ament_target_dependencies(handeye_calibration rclcpp rosbag2 rosbag2_cpp px4_msgs) #lidar_slam)
#target_link_libraries(handeye_calibration ${PCL_LIBRARIES})

add_executable(timesync_check src/timesync_check.cpp)
ament_target_dependencies(timesync_check rclcpp px4_msgs sensor_msgs)


if(GLFW_FOUND AND OPENGL_FOUND)
    include_directories(SYSTEM ${GLFW_INCLUDE_DIRS})
    add_executable(visualization_node src/visualization_node.cpp src/glad.c)
    ament_target_dependencies(visualization_node rclcpp sensor_msgs nav_msgs tf2_ros px4_msgs)
    target_link_libraries(visualization_node ${OPENGL_LIBRARIES} ${GLFW_STATIC_LIBRARIES} ${PCL_LIBRARIES})
    install(TARGETS visualization_node DESTINATION lib/${PROJECT_NAME} )


    add_executable(stereo_matching_node src/stereo_matching_node.cpp src/glad.c)
    ament_target_dependencies(stereo_matching_node rclcpp sensor_msgs nav_msgs tf2_ros px4_msgs)
    target_link_libraries(stereo_matching_node ${OPENGL_LIBRARIES} ${GLFW_STATIC_LIBRARIES} ${PCL_LIBRARIES})
    install(TARGETS stereo_matching_node DESTINATION lib/${PROJECT_NAME} )

endif()



if (BUILD_TESTING)
    enable_testing()

    find_package(ament_lint_auto REQUIRED)
    find_package(ament_cmake_gtest CONFIG REQUIRED)

    add_executable(lidar_slam_test test/lidar_slam_test.cpp)
    add_executable(g2o_test test/g2o_test.cpp)
    add_executable(kalman_filter_test test/kalman_filter_test.cpp src/kalman_filter.cpp)

    ament_target_dependencies(lidar_slam_test rclcpp sensor_msgs tf2 tf2_ros nav_msgs pcl_conversions)

    target_link_libraries(lidar_slam_test lidar_slam gtest gtest_main pthread)
    target_link_libraries(g2o_test g2o::core g2o::types_icp g2o::types_sba g2o::solver_cholmod gtest gtest_main pthread OpenMP::OpenMP_CXX)
    target_link_libraries(kalman_filter_test gtest gtest_main pthread)

    add_test(NAME lidar_slam_test COMMAND lidar_slam_test)
    add_test(NAME g2o_test COMMAND g2o_test)
    add_test(NAME kalman_filter_test COMMAND kalman_filter_test)

    # the following line skips the linter which checks for copyrights
    # uncomment the line when a copyright and license is not present in all source files
    #set(ament_cmake_copyright_FOUND TRUE)
    # the following line skips cpplint (only works in a git repo)
    # uncomment the line when this package is not in a git repo
    #set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif ()

ament_package()
