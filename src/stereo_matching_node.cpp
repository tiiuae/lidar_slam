// clang-format off
#include <lidar_slam/glad.h> // this include must go first
// clang-format on
#include <array>
#include <Eigen/Core>
#include <GLFW/glfw3.h>
#include <boost/format.hpp>
#include <rclcpp/rclcpp.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <px4_msgs/msg/vehicle_odometry.hpp>

///  Vertex-shader for computation 1 slice of cost volume
const std::string cost_vertex =
    "#version 130\n\r"
    "attribute vec2 UV;\n\r"
    "varying vec2 uv;\n\r"
    "uniform float width;\n\r"
    "uniform float height;\n\r"
    "void main()\n\r"
    "{\n\r"
    "    uv = UV;\n\r"
    "    float u = (UV.x / width) * 2.F - 1.F;\n\r"
    "    float v = 1.F - (UV.y / height) * 2.F;\n\r"
    "    gl_Position = vec4(u, v, 0.0, 1.0);\n\r"
    "}\n\r";

///  Fragment-shader for computation 1 slice of cost volume
std::string cost_frag =
    "#version 130\n\r"
    "varying vec2 uv;\n\r"
    "uniform float width;\n\r"
    "uniform float height;\n\r"
    "uniform float Z;\n\r"
    "uniform mat4 C2C1inv;\n\r"
    "uniform sampler2D MainTexture;\n\r"
    "uniform sampler2D TargetTexture;\n\r"
    "uniform sampler2D MainLookup;\n\r"
    "uniform sampler2D TargetLookup;\n\r"
    "void main()\n\r"
    "{\n\r"
    "   vec2 uv1 = vec2(uv.x/width, uv.y/height);\n\r" // texture sampler require 0..1 indexing
    "   vec4 uvz1 = vec4(uv.x*Z, uv.y*Z, Z, 1.0);\n\r" // but here we need normal coordinates
    "   vec4 uvz2 = C2C1inv*uvz1;\n\r"
    "   vec2 uv2 = vec2((uvz2.x/uvz2.z)/width, (uvz2.y/uvz2.z)/height);\n\r"
    "   vec2 UV1 = texture(MainLookup, uv1).xy;\n\r" // obtain undistorted UV coordinate
    "   vec2 UV2 = texture(TargetLookup, uv2).xy;\n\r"
    "   float Value1 = texture(MainTexture, UV1).r;\n\r"
    "   float Value2 = texture(TargetTexture, UV2).r;\n\r"
    "   float Avg1 = textureLod(MainTexture, UV1, 4).r;\n\r"
    "   float Avg2 = textureLod(TargetTexture, UV2, 4).r;\n\r"
    "   float AD1 = abs(Value1 - Value2);\n\r"
    "   float ZAD2 = abs((Value1 - Avg1) - (Value2 - Avg2));\n\r"
    "   AD1 = AD1 > 0.05 ? 0.05 : AD1;\n\r"
    "   ZAD2 = ZAD2 > 0.05 ? 0.05 : ZAD2;\n\r"
    "   float SAD = AD1 + ZAD2;\n\r"
    "   gl_FragColor = vec4(SAD, SAD, SAD, 1.0);\n\r"
    "}\n\r";

std::string main_vertex =
    "#version 130\n\r"
    "attribute vec2 UV;\n\r"
    "varying vec2 uv;\n\r"
    "uniform float width;\n\r"
    "uniform float height;\n\r"
    "void main()\n\r"
    "{\n\r"
    "    uv = vec2(1.0 - UV.x/width, 1.0 - UV.y/height);\n\r"
    "    float u = 1.F - (UV.x / width) * 2.F;\n\r"
    "    float v = 1.F - (UV.y / height) * 2.F;\n\r"
    "    gl_Position = vec4(u, v, 0.0, 1.0);\n\r"
    "}\n\r";

std::string main_frag =
    "#version 130\n\r"
    "varying vec2 uv;\n\r"
    "uniform sampler2D CostTexture;\n\r"
    "uniform float label;\n\r"
    "uniform float width;\n\r"
    "uniform float height;\n\r"
    "void main()\n\r"
    "{\n\r"
    "   float cost = textureLod(CostTexture, uv, 1.5).r;\n\r"
    "   gl_FragColor = vec4(label, label, label, 1.0);\n\r"
    //"   gl_FragColor = cost;\n\r"
    "   gl_FragDepth = cost;\n\r"
    "}\n\r";

using namespace pcl;

class StereoMatchingNode : public rclcpp::Node
{
  public:
    using ImageMsg = sensor_msgs::msg::Image ;

    StereoMatchingNode()
        : Node("stereo_matching_node"),
          left_subscriber_{},
          right_subscriber_{},
          rotation_factor_(0.001),
          travel_factor_(1)
    {
        declare_parameter<std::string>("left_image_topic", "/left/image");
        declare_parameter<std::string>("right_image_topic", "/right/image");
        declare_parameter<int>("camera_width", 1280);
        declare_parameter<int>("camera_height", 720);

        declare_parameter<float>("left_fx", 868.384133);
        declare_parameter<float>("left_fy", 868.491962);
        declare_parameter<float>("left_ox", 630.236959);
        declare_parameter<float>("left_oy", 363.474860);
        declare_parameter<float>("left_k1", 9.516443e-02);
        declare_parameter<float>("left_k2", -2.050567e-01);

        declare_parameter<float>("right_fx", 867.793500);
        declare_parameter<float>("right_fy", 867.637403);
        declare_parameter<float>("right_ox", 625.146542);
        declare_parameter<float>("right_oy", 363.586825);
        declare_parameter<float>("right_k1", 8.247173e-02);
        declare_parameter<float>("right_k2", -1.690720e-01);

        const std::string left_image_topic = get_parameter("left_image_topic").as_string();
        const std::string right_image_topic = get_parameter("right_image_topic").as_string();

        camera_width_ = get_parameter("camera_width").as_int();
        camera_height_ = get_parameter("camera_height").as_int();

        rclcpp::QoS qos(5);
        qos = qos.best_effort();

        left_subscriber_ = create_subscription<ImageMsg>(
            left_image_topic, qos, std::bind(&StereoMatchingNode::GrabLeftImage, this, std::placeholders::_1));
        right_subscriber_ = create_subscription<ImageMsg>(
            right_image_topic, qos, std::bind(&StereoMatchingNode::GrabRightImage, this, std::placeholders::_1));

        opengl_thread_running_ = true;
        opengl_thread_ = std::thread(&StereoMatchingNode::OpenGlThread, this);

        RCLCPP_INFO(get_logger(), "StereoVisualizationNode successfully initialized");
    }

    ~StereoMatchingNode()
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
    rclcpp::Subscription<ImageMsg>::SharedPtr left_subscriber_;
    rclcpp::Subscription<ImageMsg>::SharedPtr right_subscriber_;

    ImageMsg::SharedPtr latest_left_;
    ImageMsg::SharedPtr latest_right_;

    std::mutex latest_left_mutex_;
    std::mutex latest_right_mutex_;

    std::thread opengl_thread_;
    std::atomic<bool> opengl_thread_running_;
    float travel_factor_;
    float rotation_factor_;
    int camera_width_;
    int camera_height_;

    void GrabLeftImage(const ImageMsg::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(latest_left_mutex_);
        latest_left_ = msg;
    }

    void GrabRightImage(const ImageMsg::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(latest_right_mutex_);
        latest_right_ = msg;
    }

    void OpenGlThread()
    {
        const float left_fx = get_parameter("left_fx").as_double();
        const float left_fy = get_parameter("left_fy").as_double();
        const float left_ox = get_parameter("left_ox").as_double();
        const float left_oy = get_parameter("left_oy").as_double();
        const float left_k1 = get_parameter("left_k1").as_double();
        const float left_k2 = get_parameter("left_k2").as_double();

        const float right_fx = get_parameter("right_fx").as_double();
        const float right_fy = get_parameter("right_fy").as_double();
        const float right_ox = get_parameter("right_ox").as_double();
        const float right_oy = get_parameter("right_oy").as_double();
        const float right_k1 = get_parameter("right_k1").as_double();
        const float right_k2 = get_parameter("right_k2").as_double();


        ImageMsg::SharedPtr left{}, right{};
        size_t frame_id = 0;

        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

        GLFWwindow* window = OpenGLWindow(camera_width_, camera_height_);
        checkGLError("Setting OpenGL Window");

        // Set the screen with dark-blue, so we know app has started
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClearColor(0.0, 0.1, 0.2, 1.0);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glfwSwapBuffers(window);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glfwSwapBuffers(window);

        GLuint VertexArrayID;
        glGenVertexArrays(1, &VertexArrayID);
        glBindVertexArray(VertexArrayID);

        checkGLError("Cost Framebuffer ");

        GLuint costFramebufferID{};
        glGenFramebuffers(1, &costFramebufferID);
        glBindFramebuffer(GL_FRAMEBUFFER, costFramebufferID);

        // The texture we're going to render to
        GLuint renderedTextureID;
        glGenTextures(1, &renderedTextureID);
        glBindTexture(GL_TEXTURE_2D, renderedTextureID);

        // Texture filtering
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, camera_width_, camera_height_, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
        glGenerateMipmap(GL_TEXTURE_2D);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderedTextureID, 0);

        // Set the list of draw buffers.
        GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
        glDrawBuffers(1, DrawBuffers);
        if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            std::cout << "Cannot create Framebuffer" << std::endl;
            return;
        }

        checkGLError("Setting Up");

        Eigen::Matrix4f K1{};
        K1.row(0) << left_fx, 0, left_ox, 0;
        K1.row(1) << 0, left_fy, left_oy, 0;
        K1.row(2) << 0,0,1,0;
        K1.row(3) << 0,0,0,1;

        Eigen::Matrix4f K2{};
        K2.row(0) << right_fx, 0, right_ox, 0;
        K2.row(1) << 0, right_fy, right_oy, 0;
        K2.row(2) << 0,0,1,0;
        K2.row(3) << 0,0,0,1;

        Eigen::Matrix3f R1{};
        R1.row(0) << 9.9873950099709e-01, 7.1940894402835e-03, -4.9675489179424e-02 ;
        R1.row(1) << -7.6195586671084e-03, 9.9993584919744e-01, -8.3809197294464e-03 ;
        R1.row(2) << 4.9612009370801e-02, 8.7488608926038e-03, 9.9873024684310e-01;

        Eigen::Vector3f T1{2.003040e-01, -4.529036e-03, 4.621667e-03};

        Eigen::Matrix4f C1 = Eigen::Matrix4f::Identity();
        C1.block(0,0,3,3) = R1;
        C1.block(0,3,3,1) = T1;
        Eigen::Matrix4f C2C1inv = K2 * (K1 * C1).inverse();

        const GLuint costProgID = LoadProgramFromCode(cost_vertex, cost_frag, "");
        checkGLError("Loading Cost Shader Program");
        glUseProgram(costProgID);
        glUniformMatrix4fv(glGetUniformLocation(costProgID, "C2C1inv"), 1, GL_FALSE, C2C1inv.data());
        glUniform1f(glGetUniformLocation(costProgID, "width"), float(camera_width_));
        glUniform1f(glGetUniformLocation(costProgID, "height"), float(camera_height_));

        const GLuint mainProgID = LoadProgramFromCode(main_vertex, main_frag, "");
        checkGLError("Loading Main Shader Program");
        glUseProgram(mainProgID);
        glUniform1f(glGetUniformLocation(mainProgID, "width"), float(camera_width_));
        glUniform1f(glGetUniformLocation(mainProgID, "height"), float(camera_height_));

        checkGLError("Setting Textures");

        GLuint lookupLeftID = prepareLookupOpenGL(left_fx, left_fy, left_ox, left_ox, left_k1, left_k2,
                                               camera_width_, camera_height_, GL_LINEAR_MIPMAP_LINEAR);

        GLuint lookupRightID = prepareLookupOpenGL(right_fx, right_fy, right_ox, right_ox, right_k1, right_k2,
                                               camera_width_, camera_height_, GL_LINEAR_MIPMAP_LINEAR);

        GLuint textureLeftID;
        glGenTextures(1, &textureLeftID);
        glBindTexture(GL_TEXTURE_2D, textureLeftID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

        GLuint textureRightID;
        glGenTextures(1, &textureRightID);
        glBindTexture(GL_TEXTURE_2D, textureRightID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

        checkGLError("Setting Buffers");

        GLuint rectangleBufferId;
        glGenBuffers(1, &rectangleBufferId);
        glBindBuffer(GL_ARRAY_BUFFER, rectangleBufferId);
        const std::vector<float> rectangleTriangles{0.F,0.F, float(camera_width_),0.F, 0.F,float(camera_height_),
                                                    float(camera_width_),0.F, float(camera_width_),float(camera_height_), 0.F,float(camera_height_)};
        glBufferData(GL_ARRAY_BUFFER, rectangleTriangles.size() * sizeof(float), &rectangleTriangles[0], GL_STATIC_DRAW);

        while (opengl_thread_running_ && !glfwWindowShouldClose(window))
        {
            //update(window);

            {
                std::lock_guard<std::mutex> lock(latest_left_mutex_);
                if(bool(latest_left_))
                {
                    left = latest_left_;
                    std::cout << "Left Image arrived:" << left->header.stamp.sec << std::endl;
                }
            }

            {
                std::lock_guard<std::mutex> lock(latest_right_mutex_);
                if(bool(latest_right_))
                {
                    right = latest_right_;
                    std::cout << "Right Image arrived:" << right->header.stamp.sec << std::endl;
                }
            }

            if (bool(left) && bool(right))
            {
                glBindTexture(GL_TEXTURE_2D, textureLeftID);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, camera_width_, camera_height_, 0, GL_RED, GL_UNSIGNED_BYTE, &left->data[0]);
                glGenerateMipmap(GL_TEXTURE_2D);

                glBindTexture(GL_TEXTURE_2D, textureRightID);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, camera_width_, camera_height_, 0, GL_RED, GL_UNSIGNED_BYTE, &right->data[0]);
                glGenerateMipmap(GL_TEXTURE_2D);

                glEnableVertexAttribArray(0);
                glBindBuffer(GL_ARRAY_BUFFER, rectangleBufferId);
                glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

                const float minZ = 0.8F;
                const float maxZ = 10.F;
                const size_t layers = 100;

                glBindFramebuffer(GL_FRAMEBUFFER, 0);
                glEnable(GL_DEPTH_TEST);
                glDepthFunc(GL_LEQUAL);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                glClearColor(0.1, 0.2, 0.3, 1.0);


                for(size_t d=0; d<layers; d++)
                {

                    // Render to texture
                    glBindFramebuffer(GL_FRAMEBUFFER, costFramebufferID);
                    //glBindFramebuffer(GL_FRAMEBUFFER, 0);
                    glViewport(0,0,camera_width_,camera_height_);
                    glClearColor(0.5, 0.7, 0.9, 1.0);
                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                    glClearDepth(1.0);
                    glUseProgram(costProgID);

                    glActiveTexture(GL_TEXTURE0);
                    glBindTexture(GL_TEXTURE_2D, lookupLeftID);
                    glUniform1i(glGetUniformLocation(costProgID, "MainLookup"), 0);

                    glActiveTexture(GL_TEXTURE1);
                    glBindTexture(GL_TEXTURE_2D, lookupRightID);
                    glUniform1i(glGetUniformLocation(costProgID, "TargetLookup"), 1);

                    glActiveTexture(GL_TEXTURE2);
                    glBindTexture(GL_TEXTURE_2D, textureLeftID);
                    glUniform1i(glGetUniformLocation(costProgID, "MainTexture"), 2);

                    glActiveTexture(GL_TEXTURE3);
                    glBindTexture(GL_TEXTURE_2D, textureRightID);
                    glUniform1i(glGetUniformLocation(costProgID, "TargetTexture"), 3);

                    float disp = float(d)/float(layers-1);
                    float z = 1.f/(disp*(1.f/minZ - 1.f/maxZ) + 1.f/maxZ);
                    glUniform1f(glGetUniformLocation(costProgID, "Z"), z);

                    glDrawArrays(GL_TRIANGLES, 0, 6);


                    // Render to screen
                    glBindFramebuffer(GL_FRAMEBUFFER, 0);
                    glEnable(GL_DEPTH_TEST);
                    glDepthFunc(GL_LEQUAL);
                    glUseProgram(mainProgID);

                    glActiveTexture(GL_TEXTURE0);
                    glBindTexture(GL_TEXTURE_2D, renderedTextureID);
                    glGenerateMipmap(GL_TEXTURE_2D);

                    glUniform1i(glGetUniformLocation(mainProgID, "CostTexture"), 0);
                    glUniform1f(glGetUniformLocation(mainProgID, "label"), disp);

                    glDrawArrays(GL_TRIANGLES, 0, 6);
                }

                glDisableVertexAttribArray(0);


                glfwSwapBuffers(window);
            }



            glfwPollEvents();
            frame_id++;
        }

        glDeleteProgram(costProgID);
        glDeleteBuffers(1, &rectangleBufferId);
        glDeleteTextures(1, &textureLeftID);
        glDeleteTextures(1, &textureRightID);
        glDeleteTextures(1, &lookupLeftID);
        glDeleteTextures(1, &lookupRightID);
        glDeleteFramebuffers(1, &costFramebufferID);
        glDeleteTextures(1, &renderedTextureID);
        glDeleteVertexArrays(1, &VertexArrayID);

        glfwTerminate();
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

    static GLuint prepareLookupOpenGL(const float camFx,
                                      const float camFy,
                                      const float camOx,
                                      const float camOy,
                                      const float opticalK1,
                                      const float opticalK2,
                                      const int width,
                                      const int height,
                                      const GLint minFilter = GL_LINEAR)
    {
        std::unique_ptr<float[]> lookup (new float[width * height * 3]);
        float * const lookupTable = lookup.get();
        for (int yi = 0; yi < height; yi++) {
            const long row = yi * width * 3;
            for (int xi = 0; xi < width; xi++) {
                float x1 = (xi - camOx) / camFx;
                float y1 = (yi - camOy) / camFy;
                const float r1 = (x1 * x1 + y1 * y1);
                x1 = x1 * (1 + opticalK1 * r1 + opticalK2 * r1 * r1);
                y1 = y1 * (1 + opticalK1 * r1 + opticalK2 * r1 * r1);
                lookupTable[row + xi * 3] = (x1 * camFx + camOx) / float(width);
                lookupTable[row + xi * 3 + 1] = (y1 * camFy + camOy) / float(height);
                lookupTable[row + xi * 3 + 2] = 0;
            }
        }

        GLuint lookupTextureID;
        glGenTextures(1, &lookupTextureID);

        // "Bind" the newly created texture : all future texture functions will modify this texture
        glBindTexture(GL_TEXTURE_2D, lookupTextureID);

        // Texture filtering
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
        if (minFilter == GL_LINEAR_MIPMAP_LINEAR) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, lookupTable);
            glGenerateMipmap(GL_TEXTURE_2D);
        } else {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, lookupTable);
        }

        return lookupTextureID;
    }
};

int main(int argc, char* argv[])
{
    int exitCode = 0;

    rclcpp::init(argc, argv);

    try
    {
        auto node = std::make_shared<StereoMatchingNode>();
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
