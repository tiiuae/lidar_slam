// clang-format off
#include <lidar_slam/glad.h> // this include must go first
// clang-format on
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <lidar_slam/helpers.h>
#include <GLFW/glfw3.h>
#include <boost/format.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>

std::string vertex =
    "#version 120\n\r"
    "attribute vec3 RGB;\n\r"
    "attribute vec3 XYZ;\n\r"
    "varying vec3 rgb;\n\r"
    "uniform mat4 C1;\n\r"
    "uniform mat4 K;\n\r"
    "void main()\n\r"
    "{\n\r"
    "    rgb = RGB;\n\r"
    "    vec4 xyz1 = vec4(XYZ.x, XYZ.y, XYZ.z, 1.0);\n\r"
    "    vec4 uvz = K * C1 * xyz1;\n\r"
    "    vec2 uv = vec2(uvz.x / uvz.z, uvz.y / uvz.z);\n\r"
    "    float u = 2.0 * uv.x / 1024.0 - 1.0;\n\r"
    "    float v = 1.0 - 2.0 * uv.y / 768.0;\n\r"
    "    gl_Position = vec4(u, v, 0.0, 1.0);\n\r"
    "    gl_PointSize = 3.0;\n\r"
    "}\n\r";

std::string frag =
    "#version 120\n\r"
    "varying vec3 rgb;\n\r"
    "void main()\n\r"
    "{\n\r"
    "    gl_FragColor = vec4(rgb.b, rgb.g, rgb.r, 1.0);\n\r"
    "}\n\r";
using namespace lidar_slam;

class VisualizationNode : public rclcpp::Node
{
  public:
    using PointCloudMsg = sensor_msgs::msg::PointCloud2;
    using OdometryMsg = nav_msgs::msg::Odometry;
    using TransformStamped = geometry_msgs::msg::TransformStamped;
    using Point = pcl::PointXYZRGB;
    using PointCloud = pcl::PointCloud<Point>;

    VisualizationNode() : Node("visualization_node"), m_travelSpeed(0.5), m_rotationSpeed(0.001)
    {
        declare_parameter<std::string>("cloud_topic", "/camera/depth/color/points");

        const std::string cloud_topic = get_parameter("cloud_topic").as_string();
        cloud_subscriber_ = create_subscription<PointCloudMsg>(
            cloud_topic, 10, std::bind(&VisualizationNode::GrabPointCloud, this, std::placeholders::_1));

        odometry_subscriber_ = create_subscription<OdometryMsg>("/odom", 10, std::bind(&VisualizationNode::GrabOdometry, this, std::placeholders::_1));

        opengl_thread_running_ = true;
        opengl_thread_ = std::thread(&VisualizationNode::OpenGlThread, this);
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

    //PointCloud::Ptr latest_cloud_;
    PointCloudMsg::SharedPtr latest_cloud_;
    std::mutex latest_cloud_mutex_;
    OdometryMsg::SharedPtr latest_odometry_;
    std::mutex latest_odometry_mutex_;

    std::thread opengl_thread_;
    std::atomic<bool> opengl_thread_running_;
    float m_travelSpeed;
    float m_rotationSpeed;

    Eigen::Isometry3f m_pose;
    double m_lastTime = glfwGetTime();

    void GrabPointCloud(const PointCloudMsg::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(latest_cloud_mutex_);
        latest_cloud_ = msg;
    }

    void GrabOdometry(const OdometryMsg::SharedPtr msg)
    {
        std::cout << "Odometry Arrived: stamp=" << msg->header.stamp.sec << std::endl;
        std::lock_guard<std::mutex> lock(latest_odometry_mutex_);
        latest_odometry_ = msg;
    }

    void OpenGlThread()
    {
        //PointCloud::Ptr cloud;
        PointCloudMsg::SharedPtr cloud;
        OdometryMsg::SharedPtr odom;
        size_t frameId = 0;
        m_pose = Helpers::lookAt({0, 0, 0}, {1, 0, 0}, {0, 1, 0});

        // "Intrinsics"
        Eigen::Matrix4f intrinsic;
        intrinsic.row(0) << 600, 0, 768, 0;
        intrinsic.row(1) << 0, 600, 384, 0;
        intrinsic.row(2) << 0, 0, 1, 0;
        intrinsic.row(3) << 0, 0, 0, 1;

        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

        GLFWwindow* window = openWindow(1024, 768);
        checkGLError("Setting OpenGL Window");
        // Set the screen with dark-blue, so we know app has started
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClearColor(0.0, 0.1, 0.2, 1.0);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glClear(GL_COLOR_BUFFER_BIT);
        glfwSwapBuffers(window);
        glClear(GL_COLOR_BUFFER_BIT);
        glfwSwapBuffers(window);

        // Main camera-matrix
        m_pose = Eigen::Isometry3f::Identity();

        const Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();
        const GLuint mainProgID = loadProgramFromCode(vertex, frag, "");
        checkGLError("Loading Main Shader Program");
        glUseProgram(mainProgID);
        glUniformMatrix4fv(glGetUniformLocation(mainProgID, "C1"), 1, GL_FALSE, identity.data());
        glUniformMatrix4fv(glGetUniformLocation(mainProgID, "K"), 1, GL_FALSE, intrinsic.data());

        checkGLError("Setting Buffers and Textures");

        GLuint cloudBufferID;
        glGenBuffers(1, &cloudBufferID);

        while (opengl_thread_running_ && !glfwWindowShouldClose(window))
        {  // && glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS)
            {
                std::lock_guard<std::mutex> lock(latest_cloud_mutex_);
                cloud = latest_cloud_;
            }
            {
                std::lock_guard<std::mutex> lock(latest_odometry_mutex_);
                odom = latest_odometry_;
            }


            checkGLError((boost::format("OpenGL FRAME %d status: ") % frameId).str());
            std::cout << std::endl;


            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            update(window);

            if (bool(cloud) && (cloud->width * cloud->height > 0))
            {
                if(bool(odom))
                {
                    Eigen::Isometry3d odometry{};
                    odometry.prerotate(Helpers::Convert(odom->pose.pose.orientation));
                    odometry.pretranslate(Helpers::Convert(odom->pose.pose.position));

                    const Eigen::Matrix4f odom_inv = odometry.inverse().matrix().cast<float>();
                    const Eigen::Matrix4f pose = m_pose * odom_inv;
                    glUniformMatrix4fv(glGetUniformLocation(mainProgID, "C1"), 1, GL_FALSE, pose.data());
                }
                else
                {
                    const Eigen::Matrix4f pose = m_pose.matrix();
                    glUniformMatrix4fv(glGetUniformLocation(mainProgID, "C1"), 1, GL_FALSE, pose.data());
                }

                const std::size_t length = cloud->width * cloud->height;
                const std::size_t stride = cloud->point_step;
                const float* data = (float*)(&cloud->data[0]);
                const unsigned char* rgb = (unsigned char*)(&data[4]);

                std::cout << " length=" << length << "; stride=" << stride << "; overall size=" << cloud->data.size();
                std::cout << " (" << data[0] << "," << data[1] << "," << data[2] << "," << data[3] << ");";
                std::cout << " RGB:(" << int(rgb[0]) << "," << int(rgb[1]) << "," << int(rgb[2])  << ")" << std::endl;

                glUseProgram(mainProgID);
                glEnableVertexAttribArray(0);
                glEnableVertexAttribArray(1);
                glBindBuffer(GL_ARRAY_BUFFER, cloudBufferID);

                glBufferData(GL_ARRAY_BUFFER, length * stride, data, GL_DYNAMIC_DRAW);

                glVertexAttribPointer(0, 3, GL_UNSIGNED_BYTE, GL_TRUE, stride,  (GLvoid*)(4 * sizeof(GL_FLOAT)));
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, nullptr);

                glDrawArrays(GL_POINTS, 0, GLsizei(length));
                glDisableVertexAttribArray(1);
                glDisableVertexAttribArray(0);
            }
            else
            {
                std::cout << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }

            glfwSwapBuffers(window);
            glfwPollEvents();
            frameId++;
        }

        glDeleteProgram(mainProgID);
        glDeleteBuffers(1, &cloudBufferID);


        glfwTerminate();
    }

    void update(GLFWwindow* window)
    {
        // read all OpenGL events, happened just prior to call
        glfwPollEvents();

        int screen_width, screen_height;
        glfwGetWindowSize(window, &screen_width, &screen_height);

        double currentTime = glfwGetTime();
        auto deltaTime = float(currentTime - m_lastTime);

        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS)
        {
            // Get mouse cursor m_position
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);

            // if mouse outside of the OpenGL window - ignore the motion
            xpos = abs(xpos - screen_width / 2.0) < 2.0 ? screen_width / 2.0 : xpos;
            ypos = abs(ypos - screen_height / 2.0) < 2.0 ? screen_height / 2.0 : ypos;
            // Reset mouse m_position for next frame
            glfwSetCursorPos(window, screen_width / 2.0, screen_height / 2.0);

            const float horizontal = m_rotationSpeed * (screen_width / 2.f - float(xpos));
            const float vertical = m_rotationSpeed * (screen_height / 2.f - float(ypos));

            Eigen::AngleAxisf rollAngle(horizontal, Eigen::Vector3f::UnitY());
            Eigen::AngleAxisf pitchAngle(-vertical, Eigen::Vector3f::UnitX());

            m_pose.prerotate(rollAngle);
            m_pose.prerotate(pitchAngle);
        }

        float x = 0, y = 0, z = 0;
        // Move backward
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS || glfwGetKey(window, 'W') == GLFW_PRESS)
        {
            x = deltaTime * m_travelSpeed;
        }
        // Move backward
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS || glfwGetKey(window, 'S') == GLFW_PRESS)
        {
            x = -deltaTime * m_travelSpeed;
        }
        // Strafe right
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS || glfwGetKey(window, 'D') == GLFW_PRESS)
        {
            y = deltaTime * m_travelSpeed;
        }
        // Strafe left
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS || glfwGetKey(window, 'A') == GLFW_PRESS)
        {
            y = -deltaTime * m_travelSpeed;
        }

        // Move UP
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_PAGE_UP) == GLFW_PRESS)
        {
            z = deltaTime * m_travelSpeed;
        }

        // Move Down
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS ||
            glfwGetMouseButton(window, GLFW_KEY_PAGE_DOWN) == GLFW_PRESS)
        {
            z = -deltaTime * m_travelSpeed;
        }

        m_pose.pretranslate(Eigen::Vector3f({-y, z, -x}));

        // For the next frame, the "last time" will be "now"
        m_lastTime = currentTime;
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

    static GLFWwindow* openWindow(const int width,
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

    static GLuint compileShader(const std::string& shaderCode, const GLuint shaderType)
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

    static GLuint compileProgram(const GLuint vertexShaderID,
                                 const GLuint fragmentShaderID,
                                 const GLuint geometryShaderID = 0)
    {
        // Link the program
        GLuint ProgramID = glCreateProgram();
        GLint result = GL_FALSE;
        int logLength;

        std::cout << "Linking the program " << ProgramID;
        glAttachShader(ProgramID, vertexShaderID);
        glAttachShader(ProgramID, fragmentShaderID);
        if (geometryShaderID > 0)
        {
            glAttachShader(ProgramID, geometryShaderID);
        }
        glLinkProgram(ProgramID);

        // Check the program
        glGetProgramiv(ProgramID, GL_LINK_STATUS, &result);
        glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &logLength);
        if (!result)
        {
            std::vector<char> errorMessage(logLength + 1);
            glGetProgramInfoLog(ProgramID, logLength, nullptr, &errorMessage[0]);
            throw std::runtime_error(&errorMessage[0]);
        }
        return ProgramID;
    }

    static GLuint loadProgramFromCode(const std::string& vertexCode,
                                      const std::string& fragmentCode,
                                      const std::string geometryCode = std::string(""))
    {
        // Compile the shaders
        std::cout << "Compiling Vertex Shader";
        GLuint vertexShaderID = compileShader(vertexCode, GL_VERTEX_SHADER);
        std::cout << "Compiling Fragment Shader";
        GLuint fragmentShaderID = compileShader(fragmentCode, GL_FRAGMENT_SHADER);
        GLuint geometryShaderID = 0;
        if (!geometryCode.empty())
        {
            std::cout << "Compiling Geometry Shader";
            geometryShaderID = compileShader(geometryCode, GL_GEOMETRY_SHADER);
        }

        GLuint ProgramID = compileProgram(vertexShaderID, fragmentShaderID, geometryShaderID);

        glDeleteShader(vertexShaderID);
        glDeleteShader(fragmentShaderID);
        if (!geometryCode.empty())
        {
            glDeleteShader(geometryShaderID);
        }
        return ProgramID;
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
