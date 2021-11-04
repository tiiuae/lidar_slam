// clang-format off
#include <lidar_slam/glad.h> // this include must go first
// clang-format on
#include <array>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <lidar_slam/ros_helpers.h>
#include <lidar_slam/helpers.h>
#include <GLFW/glfw3.h>
#include <boost/format.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <lidar_slam/helpers.h>

std::string main_vertex =
    "#version 120\n\r"
    "attribute vec3 RGB;\n\r"
    "attribute vec3 XYZ;\n\r"
    "varying vec3 rgb;\n\r"
    "uniform mat4 C1;\n\r"
    "uniform mat4 K;\n\r"
    "uniform float scale;\n\r"
    "void main()\n\r"
    "{\n\r"
    "    rgb = RGB;\n\r"
    "    vec4 xyz1 = vec4(XYZ.x * scale, XYZ.y * scale, XYZ.z * scale, 1.0) ;\n\r"
    "    gl_Position = K * C1 * xyz1;\n\r"
    "    gl_PointSize = 3.0;\n\r"
    "}\n\r";

std::string main_frag =
    "#version 120\n\r"
    "varying vec3 rgb;\n\r"
    "void main()\n\r"
    "{\n\r"
    "    gl_FragColor = vec4(rgb.b, rgb.g, rgb.r, 1.0);\n\r"
    "}\n\r";

using namespace lidar_slam;
using namespace pcl;

class VisualizationNode : public rclcpp::Node
{
  public:
    using PointCloudMsg = sensor_msgs::msg::PointCloud2;
    using OdometryMsg = nav_msgs::msg::Odometry;
    using TransformStamped = geometry_msgs::msg::TransformStamped;
    using Point = pcl::PointXYZ;
    using PointCloud = pcl::PointCloud<Point>;
    using VehicleOdometry = px4_msgs::msg::VehicleOdometry;

    VisualizationNode()
        : Node("visualization_node"),
          cloud_subscriber_{},
          base_frame_{},
          map_frame_{},
          odom_frame_{},
          rotation_factor_(0.001),
          travel_factor_(1),
          clock_(RCL_SYSTEM_TIME),
          tf_buffer_(std::make_shared<rclcpp::Clock>(clock_)),
          tf_listener_(tf_buffer_, true)
    {
        declare_parameter<std::string>("cloud_topic", "/camera/depth/color/points");
        declare_parameter<std::string>("base_frame_id", "base_link");
        declare_parameter<std::string>("map_frame_id", "map");
        declare_parameter<std::string>("odom_frame_id", "odom");
        declare_parameter<std::string>("sensor_frame_id", "camera_depth_optical_frame");
        declare_parameter<int>("screen_width", 1024);
        declare_parameter<int>("screen_height", 768);

        const std::string cloud_topic = get_parameter("cloud_topic").as_string();
        base_frame_ = get_parameter("base_frame_id").as_string();
        map_frame_ = get_parameter("map_frame_id").as_string();
        odom_frame_ = get_parameter("odom_frame_id").as_string();
        sensor_frame_ = get_parameter("sensor_frame_id").as_string();
        screen_width_ = get_parameter("screen_width").as_int();
        screen_height_ = get_parameter("screen_height").as_int();

        rclcpp::QoS qos(5);
        qos = qos.best_effort();

        cloud_subscriber_ = create_subscription<PointCloudMsg>(
            cloud_topic, qos, std::bind(&VisualizationNode::GrabPointCloud, this, std::placeholders::_1));

        odometry_subscriber_ = create_subscription<OdometryMsg>(
            std::string("/odom"), qos, std::bind(&VisualizationNode::GrabOdometry, this, std::placeholders::_1));

        mapping_subscriber_ = create_subscription<OdometryMsg>(
            std::string("/map"), qos, std::bind(&VisualizationNode::GrabMapping, this, std::placeholders::_1));

        px4_odometry_subscriber_ = create_subscription<VehicleOdometry>(
            std::string("/default/fmu/vehicle_local_position/out"),
            qos,
            std::bind(&VisualizationNode::GrabPx4Odometry, this, std::placeholders::_1));

        opengl_thread_running_ = true;
        opengl_thread_ = std::thread(&VisualizationNode::OpenGlThread, this);

        RCLCPP_INFO(get_logger(), "VisualizationNode successfully initialized");
    }

    ~VisualizationNode()
    {
        if (opengl_thread_running_ == true)
        {
            opengl_thread_running_ = false;
        }

        if (opengl_thread_.joinable())
        {
            opengl_thread_.join();
        }
    }

  private:
    rclcpp::Subscription<PointCloudMsg>::SharedPtr cloud_subscriber_;
    rclcpp::Subscription<OdometryMsg>::SharedPtr odometry_subscriber_;
    rclcpp::Subscription<OdometryMsg>::SharedPtr mapping_subscriber_;
    rclcpp::Subscription<VehicleOdometry>::SharedPtr px4_odometry_subscriber_;
    std::string base_frame_, map_frame_, odom_frame_, sensor_frame_;
    rclcpp::Clock clock_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    PointCloudMsg::SharedPtr latest_cloud_;
    boost::optional<Eigen::Matrix4f> latest_corner_;
    std::mutex latest_cloud_mutex_;

    OdometryMsg::SharedPtr latest_odometry_;
    std::vector<Eigen::Isometry3d> odometries_;
    std::mutex latest_odometry_mutex_;

    OdometryMsg::SharedPtr latest_mapping_;
    std::mutex latest_mapping_mutex_;

    VehicleOdometry::SharedPtr latest_px4_odometry_;
    std::mutex latest_px4_odometry_mutex_;

    std::thread opengl_thread_;
    std::atomic<bool> opengl_thread_running_;
    float travel_factor_;
    float rotation_factor_;
    int screen_width_;
    int screen_height_;

    Eigen::Isometry3f main_pose_;
    double last_time_ = glfwGetTime();

    void GrabPointCloud(const PointCloudMsg::SharedPtr msg)
    {
        if (bool(latest_cloud_))
        {
            PointCloud::Ptr cloud(new PointCloud());
            {
                std::lock_guard<std::mutex> lock(latest_cloud_mutex_);
                pcl::moveFromROSMsg<Point>(*latest_cloud_, *cloud);
                latest_cloud_ = msg;
            }
            latest_corner_ = Helpers::Detect3DCorner<Point>(cloud);
        }
        else
        {
            std::lock_guard<std::mutex> lock(latest_cloud_mutex_);
            latest_cloud_ = msg;
        }
    }

    void GrabOdometry(const OdometryMsg::SharedPtr msg)
    {
        RCLCPP_INFO(get_logger(), "Odometry Arrived: " + std::to_string(msg->header.stamp.sec));

        Eigen::Isometry3d odometry = Eigen::Isometry3d::Identity();
        odometry.prerotate(RosHelpers::Convert(msg->pose.pose.orientation));
        odometry.pretranslate(RosHelpers::Convert(msg->pose.pose.position));

        if (!odometries_.empty())
        {
            const Eigen::Isometry3d previous = odometries_.at(odometries_.size() - 1);
            const Eigen::Matrix4d diff = (odometry * previous.inverse()).matrix();
            auto shiftangle = Helpers::GetAbsoluteShiftAngle(diff);
            const double new_node_min_translation = 0.1;  // [meters]
            const double new_node_after_rotation = 0.05;  // [radians]

            if (shiftangle.first > new_node_min_translation || shiftangle.second > new_node_after_rotation)
            {
                std::lock_guard<std::mutex> lock(latest_odometry_mutex_);
                odometries_.push_back(odometry);
            }
        }
        std::lock_guard<std::mutex> lock(latest_odometry_mutex_);
        latest_odometry_ = msg;
    }

    void GrabMapping(const OdometryMsg::SharedPtr msg)
    {
        RCLCPP_INFO(get_logger(), "Mapping Arrived: " + std::to_string(msg->header.stamp.sec));
        std::lock_guard<std::mutex> lock(latest_mapping_mutex_);
        latest_mapping_ = msg;
    }

    void GrabPx4Odometry(const VehicleOdometry::SharedPtr msg)
    {
        RCLCPP_INFO(get_logger(),
                    "VehicleOdometry Arrived: " + std::to_string(msg->timestamp) + " X=" + std::to_string(msg->x) +
                        " Y=" + std::to_string(msg->y) + " Z=" + std::to_string(msg->z));

        std::lock_guard<std::mutex> lock(latest_px4_odometry_mutex_);
        latest_px4_odometry_ = msg;
    }

    void OpenGlThread()
    {
        enum class GlBufferId : unsigned
        {
            Cloud,
            CoordLines,
            CornerLines,
            CoordColors,
            GrayColors,
            Size  // Keep that one the last
        };

#define GLBUFFSIZE unsigned(GlBufferId::Size)

        PointCloudMsg::SharedPtr cloud{};
        OdometryMsg::SharedPtr odom{};
        OdometryMsg::SharedPtr map{};
        VehicleOdometry::SharedPtr px4_odometry{};
        boost::optional<Eigen::Matrix4f> corner;
        size_t frame_id = 0;

        // Camera-view matrix
        main_pose_ = Helpers::lookAt({0, 0, -1}, {0, 0, 0}, {0, -1, 0});

        // Projection matrix (aka "intrinsics")
        const float aspect = float(screen_width_) / float(screen_height_);
        Eigen::Matrix4f intrinsic = Helpers::perspective(50.F, aspect, 0.01F, 10.F);

        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

        GLFWwindow* window = OpenGLWindow(screen_width_, screen_height_);
        checkGLError("Setting OpenGL Window");
        // Set the screen with dark-blue, so we know app has started
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClearColor(0.0, 0.1, 0.2, 1.0);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glPointSize(3);
        glLineWidth(3);
        glClear(GL_COLOR_BUFFER_BIT);
        glfwSwapBuffers(window);
        glClear(GL_COLOR_BUFFER_BIT);
        glfwSwapBuffers(window);

        GLuint VertexArrayID;
        glGenVertexArrays(1, &VertexArrayID);
        glBindVertexArray(VertexArrayID);

        const GLuint mainProgID = LoadProgramFromCode(main_vertex, main_frag, "");
        checkGLError("Loading Main Shader Program");
        glUseProgram(mainProgID);
        glUniformMatrix4fv(glGetUniformLocation(mainProgID, "C1"), 1, GL_FALSE, main_pose_.data());
        glUniformMatrix4fv(glGetUniformLocation(mainProgID, "K"), 1, GL_FALSE, intrinsic.data());
        glUniform1f(glGetUniformLocation(mainProgID, "scale"), 1.f);

        checkGLError("Setting Buffers and Textures");

        std::array<GLuint, GLBUFFSIZE> gl_buffer_ids{};
        glGenBuffers(GLBUFFSIZE, &gl_buffer_ids[0]);

#define GLBUFFID(Name) gl_buffer_ids[unsigned(GlBufferId::Name)]

        const std::vector<float> coord_lines{0, 0, 0, 1., 0, 0, 0, 0, 0, 0, 1., 0, 0, 0, 0, 0, 0, 1.};
        const std::vector<float> corner_lines{0, 0, 0, 1., 0, 0, 0, 0, 0, 0, -1., 0, 0, 0, 0, 0, 0, -1.};
        const std::vector<float> coord_colors{0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0};
        const std::vector<float> gray_colors{
            0.5, 0.5, 0.9, 0.5, 0.5, 0.9, 0.5, 0.9, 0.5, 0.5, 0.9, 0.5, 0.9, 0.5, 0.5, 0.9, 0.5, 0.5};
        std::vector<Eigen::Matrix4f> trajectory;
        glBindBuffer(GL_ARRAY_BUFFER, GLBUFFID(CoordLines));
        glBufferData(GL_ARRAY_BUFFER, coord_lines.size() * sizeof(float), &coord_lines[0], GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, GLBUFFID(CornerLines));
        glBufferData(GL_ARRAY_BUFFER, corner_lines.size() * sizeof(float), &corner_lines[0], GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, GLBUFFID(CoordColors));
        glBufferData(GL_ARRAY_BUFFER, coord_colors.size() * sizeof(float), &coord_colors[0], GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, GLBUFFID(GrayColors));
        glBufferData(GL_ARRAY_BUFFER, gray_colors.size() * sizeof(float), &gray_colors[0], GL_STATIC_DRAW);

        while (opengl_thread_running_ && !glfwWindowShouldClose(window))
        {
            glClear(GL_COLOR_BUFFER_BIT);
            glClear(GL_DEPTH_BUFFER_BIT);
            update(window);

            {
                std::lock_guard<std::mutex> lock(latest_cloud_mutex_);
                corner = latest_corner_;
                cloud = latest_cloud_;
            }

            {
                std::lock_guard<std::mutex> lock(latest_odometry_mutex_);
                odom = latest_odometry_;
            }
            {
                std::lock_guard<std::mutex> lock(latest_mapping_mutex_);
                map = latest_mapping_;
            }
            {
                std::lock_guard<std::mutex> lock(latest_px4_odometry_mutex_);
                px4_odometry = latest_px4_odometry_;
            }

            if (bool(cloud) && (cloud->width * cloud->height > 0))
            {
                const std::size_t length = cloud->width * cloud->height;
                const std::size_t stride = cloud->point_step;
                const float* data = (float*)(&cloud->data[0]);

                Eigen::Isometry3d mapping = Eigen::Isometry3d::Identity();
                Eigen::Isometry3d odometry = Eigen::Isometry3d::Identity();
                Eigen::Matrix4f pose = main_pose_.matrix();

                if (bool(map))
                {
                    mapping.prerotate(RosHelpers::Convert(map->pose.pose.orientation));
                    mapping.pretranslate(RosHelpers::Convert(map->pose.pose.position));
                }

                if (bool(odom))
                {

                    odometry.prerotate(RosHelpers::Convert(odom->pose.pose.orientation));
                    odometry.pretranslate(RosHelpers::Convert(odom->pose.pose.position));

                    pose = main_pose_ * mapping.matrix().cast<float>() * odometry.matrix().cast<float>();
                    glUniformMatrix4fv(glGetUniformLocation(mainProgID, "C1"), 1, GL_FALSE, pose.data());
                    glUniform1f(glGetUniformLocation(mainProgID, "scale"), 0.2f);

                    // draw the sensor origin
                    glLineWidth(4);
                    glUseProgram(mainProgID);
                    glEnableVertexAttribArray(0);
                    glBindBuffer(GL_ARRAY_BUFFER, GLBUFFID(CoordColors));
                    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

                    glEnableVertexAttribArray(1);
                    glBindBuffer(GL_ARRAY_BUFFER, GLBUFFID(CoordLines));
                    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
                    glDrawArrays(GL_LINES, 0, 6);
                    glDisableVertexAttribArray(1);
                    glDisableVertexAttribArray(0);

                    // draw trajectory
                    for (auto odom_i : odometries_)
                    {
                        const Eigen::Matrix4f pose_i =
                            main_pose_ * mapping.matrix().cast<float>() * odom_i.matrix().cast<float>();
                        glUniformMatrix4fv(glGetUniformLocation(mainProgID, "C1"), 1, GL_FALSE, pose_i.data());
                        glUniform1f(glGetUniformLocation(mainProgID, "scale"), 0.1f);

                        // draw the sensor origin
                        glLineWidth(4);
                        glUseProgram(mainProgID);
                        glEnableVertexAttribArray(0);
                        glBindBuffer(GL_ARRAY_BUFFER, GLBUFFID(GrayColors));
                        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

                        glEnableVertexAttribArray(1);
                        glBindBuffer(GL_ARRAY_BUFFER, GLBUFFID(CoordLines));
                        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
                        glDrawArrays(GL_LINES, 0, 6);
                        glDisableVertexAttribArray(1);
                        glDisableVertexAttribArray(0);
                    }
                }

                // draw all the points
                glUseProgram(mainProgID);
                glUniformMatrix4fv(glGetUniformLocation(mainProgID, "C1"), 1, GL_FALSE, pose.data());
                glUniform1f(glGetUniformLocation(mainProgID, "scale"), 1.f);
                glEnableVertexAttribArray(0);

                glBindBuffer(GL_ARRAY_BUFFER, GLBUFFID(Cloud));
                glBufferData(GL_ARRAY_BUFFER, length * stride, data, GL_DYNAMIC_DRAW);
                glVertexAttribPointer(0, 3, GL_UNSIGNED_BYTE, GL_TRUE, stride, (GLvoid*)(4 * sizeof(float)));

                glEnableVertexAttribArray(1);
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, nullptr);
                glDrawArrays(GL_POINTS, 0, GLsizei(length));
                glDisableVertexAttribArray(1);
                glDisableVertexAttribArray(0);
            }

            if (corner.has_value())
            {
                const Eigen::Matrix4f corner_pose = main_pose_ * corner.value();
                glUniformMatrix4fv(glGetUniformLocation(mainProgID, "C1"), 1, GL_FALSE, corner_pose.data());
                glUniform1f(glGetUniformLocation(mainProgID, "scale"), 0.2f);
                // draw the sensor origin
                glLineWidth(4);
                glUseProgram(mainProgID);
                glEnableVertexAttribArray(0);
                glBindBuffer(GL_ARRAY_BUFFER, GLBUFFID(CoordColors));
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

                glEnableVertexAttribArray(1);
                glBindBuffer(GL_ARRAY_BUFFER, GLBUFFID(CornerLines));
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
                glDrawArrays(GL_LINES, 0, 6);
                glDisableVertexAttribArray(1);
                glDisableVertexAttribArray(0);
            }

            //            if(bool(px4_odometry))
            //            {
            //                Eigen::Matrix4f px4_pose = Eigen::Matrix4f::Identity();
            //
            //                px4_pose(0,3) = px4_odometry->x;
            //                px4_pose(1,3) = px4_odometry->y;
            //                px4_pose(2,3) = px4_odometry->z;
            //
            //                Eigen::AngleAxisf yaw_rot(px4_odometry->heading, Eigen::Vector3f::UnitZ());
            //                px4_pose.block(0,0,3,3) = yaw_rot.toRotationMatrix();
            //
            //                px4_pose = main_pose_ * px4_pose;
            //
            //                glUniformMatrix4fv(glGetUniformLocation(mainProgID, "C1"), 1, GL_FALSE, px4_pose.data());
            //
            //                glLineWidth(4);
            //                glUseProgram(mainProgID);
            //                glEnableVertexAttribArray(0);
            //                glBindBuffer(GL_ARRAY_BUFFER, GLBUFFID(CoordColors));
            //                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
            //
            //                glEnableVertexAttribArray(1);
            //                glBindBuffer(GL_ARRAY_BUFFER, GLBUFFID(CoordLines));
            //                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
            //                glDrawArrays(GL_LINES, 0, 6);
            //                glDisableVertexAttribArray(1);
            //                glDisableVertexAttribArray(0);
            //            }

            glfwSwapBuffers(window);
            glfwPollEvents();
            frame_id++;
        }

        glDeleteProgram(mainProgID);
        glDeleteBuffers(GLBUFFSIZE, &gl_buffer_ids[0]);
        glDeleteVertexArrays(1, &VertexArrayID);

        glfwTerminate();
    }

    void update(GLFWwindow* window)
    {
        // read all OpenGL events, happened just prior to call
        glfwPollEvents();

        //        int screen_width, screen_height;
        //        glfwGetWindowSize(window, &screen_width, &screen_height);

        double current_time = glfwGetTime();
        auto delta_time = float(current_time - last_time_);

        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS)
        {
            // Get mouse cursor m_position
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);

            // if mouse outside of the OpenGL window - ignore the motion
            xpos = abs(xpos - screen_width_ / 2.0) < 2.0 ? screen_width_ / 2.0 : xpos;
            ypos = abs(ypos - screen_height_ / 2.0) < 2.0 ? screen_height_ / 2.0 : ypos;
            // Reset mouse position for next frame
            glfwSetCursorPos(window, screen_width_ / 2.0, screen_height_ / 2.0);

            const float horizontal = rotation_factor_ * (screen_width_ / 2.f - float(xpos));
            const float vertical = rotation_factor_ * (screen_height_ / 2.f - float(ypos));

            Eigen::AngleAxisf rollAngle(-horizontal, Eigen::Vector3f::UnitY());
            Eigen::AngleAxisf pitchAngle(-vertical, Eigen::Vector3f::UnitX());

            main_pose_.prerotate(rollAngle);
            main_pose_.prerotate(pitchAngle);
        }

        float x = 0, y = 0, z = 0;
        // Move forward
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS || glfwGetKey(window, 'W') == GLFW_PRESS)
        {
            z = delta_time * travel_factor_;
        }
        // Move backward
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS || glfwGetKey(window, 'S') == GLFW_PRESS)
        {
            z = -delta_time * travel_factor_;
        }
        // Strafe right
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS || glfwGetKey(window, 'D') == GLFW_PRESS)
        {
            x = -delta_time * travel_factor_;
        }
        // Strafe left
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS || glfwGetKey(window, 'A') == GLFW_PRESS)
        {
            x = delta_time * travel_factor_;
        }

        // Move UP
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_PAGE_UP) == GLFW_PRESS)
        {
            y = -delta_time * travel_factor_;
        }

        // Move Down
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS ||
            glfwGetMouseButton(window, GLFW_KEY_PAGE_DOWN) == GLFW_PRESS)
        {
            y = delta_time * travel_factor_;
        }

        main_pose_.pretranslate(Eigen::Vector3f({x, y, z}));

        // For the next frame, the "last time" will be "now"
        last_time_ = current_time;
    }

    static void checkGLError(const std::string msg)
    {
        GLenum err = glGetError();
        if (err != GL_NO_ERROR)
        {
            throw std::runtime_error((boost::format("%s. OpenGL Error 0x%04x!") % msg % err).str());
        }
        else
        {
            std::cout << msg << ": SUCCESSFUL";
        }
    }

    static GLFWwindow* OpenGLWindow(const int width,
                                    const int height,
                                    const int swapInterval = 0,
                                    std::string name = std::string("Slam App"))
    {
        // Initialise GLFW
        if (!glfwInit())
        {
            throw std::runtime_error("Failed to initialize GLFW");
        }

        GLFWwindow* window = glfwCreateWindow(width, height, name.c_str(), nullptr, nullptr);

        // Open a window and create its OpenGL context
        if (!window)
        {
            glfwTerminate();
            throw std::runtime_error("Failed to open GLFW window.");
        }

        glfwMakeContextCurrent(window);

        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        {
            throw std::runtime_error("Failed to initialize OpenGL context");
        }

        glfwSwapInterval(swapInterval);

        // Cannot use this function before GLAD initialization!
        checkGLError("OpenGL Initialization");

        return window;
    }

    static GLuint CompileShader(const std::string& shaderCode, const GLuint shaderType)
    {
        GLuint shaderID = glCreateShader(shaderType);
        GLint result = GL_FALSE;
        int logLength;

        // Compile Shader
        char const* sourcePointer = shaderCode.c_str();
        glShaderSource(shaderID, 1, &sourcePointer, nullptr);
        glCompileShader(shaderID);

        // Check Shader
        glGetShaderiv(shaderID, GL_COMPILE_STATUS, &result);
        glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &logLength);
        if (!result)
        {
            std::cout << shaderCode;
            std::vector<char> shaderErrorMessage(logLength + 1);
            glGetShaderInfoLog(shaderID, logLength, nullptr, &shaderErrorMessage[0]);
            throw std::runtime_error(&shaderErrorMessage[0]);
        }

        return shaderID;
    }

    static GLuint CompileProgram(const GLuint vertexShaderID,
                                 const GLuint fragmentShaderID,
                                 const GLuint geometryShaderID = 0)
    {
        // Link the program
        GLuint programID = glCreateProgram();
        GLint result = GL_FALSE;
        int logLength;

        std::cout << "Linking the program " << programID;
        glAttachShader(programID, vertexShaderID);
        glAttachShader(programID, fragmentShaderID);
        if (geometryShaderID > 0)
        {
            glAttachShader(programID, geometryShaderID);
        }
        glLinkProgram(programID);

        // Check the program
        glGetProgramiv(programID, GL_LINK_STATUS, &result);
        glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &logLength);
        if (!result)
        {
            std::vector<char> errorMessage(logLength + 1);
            glGetProgramInfoLog(programID, logLength, nullptr, &errorMessage[0]);
            throw std::runtime_error(&errorMessage[0]);
        }
        return programID;
    }

    static GLuint LoadProgramFromCode(const std::string& vertexCode,
                                      const std::string& fragmentCode,
                                      const std::string geometryCode = std::string(""))
    {
        // Compile the shaders
        std::cout << "Compiling Vertex Shader";
        GLuint vertexShaderID = CompileShader(vertexCode, GL_VERTEX_SHADER);
        std::cout << "Compiling Fragment Shader";
        GLuint fragmentShaderID = CompileShader(fragmentCode, GL_FRAGMENT_SHADER);
        GLuint geometryShaderID = 0;
        if (!geometryCode.empty())
        {
            std::cout << "Compiling Geometry Shader";
            geometryShaderID = CompileShader(geometryCode, GL_GEOMETRY_SHADER);
        }

        GLuint programID = CompileProgram(vertexShaderID, fragmentShaderID, geometryShaderID);

        glDeleteShader(vertexShaderID);
        glDeleteShader(fragmentShaderID);
        if (!geometryCode.empty())
        {
            glDeleteShader(geometryShaderID);
        }
        return programID;
    }
};

int main(int argc, char* argv[])
{
    int exitCode = 0;

    rclcpp::init(argc, argv);

    try
    {
        auto node = std::make_shared<VisualizationNode>();
        rclcpp::spin(node);
    }
    catch (std::runtime_error& e)
    {
        std::cout << e.what();
    }
    catch (std::exception& e)
    {
        std::cout << e.what();
    }

    rclcpp::shutdown();

    return 0;

    // Close OpenGL window and terminate GLFW
    glfwTerminate();

    return exitCode;
}
