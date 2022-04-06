#include <iostream>
#include <memory>
#include <vector>
#include "FileReader.h"
#include "RayTracer.h"
#include <glad/glad.h>
#include <glfw3.h>
#include <math.h>

double fov = 90.0;
double last_x = 0.0;
double last_y = 0.0;
double yaw = 0.0;
double pitch = 0.0;
bool lmb_hold = false;
bool first_lmb = true;
int window_width = 640, window_height = 480;

Core::model* pcamera = 0;
RayTracer::Projection proj_type = RayTracer::Projection::PERSPECTIVE;
Core::vec3 translater;

void check_btn_press(GLFWwindow* window);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xpos, double ypos);
void prepare_input(RayTracer::input& i);

void main()
{
	std::cout << "Loading..." << std::endl;
	const char* window_title = "GraphicsEngine";
	bool init_called = false;
	std::unique_ptr<Core::model> pmodel;
	RayTracer::input i;
	try
	{
		//pmodel = prepare_gltf_model_data({ "D:/Projects/C++/3DImporter/Assets/airplane/scene.gltf", Core::vec3{0.0,0.0,-3.0} });
		std::unique_ptr<Core::model> psphere = prepare_spheres();
		std::unique_ptr<Core::model> pbox = prepare_boxes();

		Core::model light;
		Core::sphere* pspheres = new Core::sphere[1];
		pspheres[0] = Core::sphere{ 2.0, Core::vec3{ 75.0, 100.0, -100.0  } };
		light.position = pspheres[0].center;
		light.surface_color = Core::vec3{ 1.0, 1.0, 0.0 };
		light.emissive_color = Core::vec3{ 1.0, 1.0, 1.0 };
		light.pshapes = pspheres;
		light.shapes_size = 1;
		light.s_type = Core::shape_type::SPHERE;
		light.m_type = Core::model_type::LIGHT;

		Core::model camera{ Core::vec3{} };
		camera.m_type = Core::model_type::CAMERA;
		pcamera = &camera;
		std::shared_ptr<std::vector<Core::model*>> pmodels(new std::vector<Core::model*>);
		//pmodels->push_back(pmodel.get());
		pmodels->push_back(psphere.get());
		pmodels->push_back(pbox.get());
		pmodels->push_back(&light);
		pmodels->push_back(&camera);
		std::unique_ptr<Core::cubemap> p = prepare_cubemap("D:/Projects/C++/3DImporter/Assets/skybox");
		RayTracer::init(pmodels, p.get(), window_width, window_height);
		init_called = true;
	}
	catch (std::exception& e)
	{
		std::cout << "Exception thrown :: "<< e.what() << std::endl;
	}
	std::cout << "Opening window" << std::endl;
	if (init_called)
	{
		const char* window_title = "GraphicsEngine";

		GLfloat square_vertices[] =
		{
			-1.0f, -1.0f, 0.0f,   1.0f, 1.0f,
			 1.0f, -1.0f, 0.0f,   0.0f, 1.0f,
			-1.0f,  1.0f, 0.0f,   1.0f, 0.0f,
			 1.0f,  1.0f, 0.0f,   0.0f, 0.0f
		};

		GLuint square_indices[] =
		{
			0, 1, 2,
			2, 3, 1
		};

		int has_compiled;

		glfwInit();
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		GLFWwindow* window = glfwCreateWindow(window_width, window_height, window_title, NULL, NULL);
		if (window == NULL)
		{
			std::cout << "Error loading window for Exercise3::easy_problem" << std::endl;
			glfwTerminate();
		}
		glfwMakeContextCurrent(window);
		gladLoadGL();
		glViewport(0, 0, window_width, window_height);

		std::string vertex_string = extract_file("tex.vert");
		std::string fragment_string = extract_file("tex.frag");
		const char* vertex_shader_source = vertex_string.c_str();
		const char* fragment_shader_source = fragment_string.c_str();

		GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertex_shader, 1, &vertex_shader_source, NULL);
		glCompileShader(vertex_shader);
		glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &has_compiled);
		if (!has_compiled)
		{
			std::cout << "Error compiling vertex shader" << std::endl;
		}

		GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragment_shader, 1, &fragment_shader_source, NULL);
		glCompileShader(fragment_shader);
		glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &has_compiled);
		if (!has_compiled)
		{
			std::cout << "Error compiling fragment shader" << std::endl;
		}

		GLuint shader_program = glCreateProgram();
		glAttachShader(shader_program, vertex_shader);
		glAttachShader(shader_program, fragment_shader);
		glLinkProgram(shader_program);
		glGetShaderiv(shader_program, GL_COMPILE_STATUS, &has_compiled);
		if (!has_compiled)
		{
			std::cout << "Error linking shader program" << std::endl;
		}
		glDeleteShader(vertex_shader);
		glDeleteShader(fragment_shader);

		GLuint VBO, VAO;
		glGenBuffers(1, &VBO);
		glGenVertexArrays(1, &VAO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(square_vertices), square_vertices, GL_STATIC_DRAW);
		glBindVertexArray(VAO);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);

		GLuint EBO;
		glGenBuffers(1, &EBO);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(square_indices), square_indices, GL_STATIC_DRAW);

		GLuint texture;
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		glfwSetCursorPosCallback(window, mouse_callback);
		glfwSetScrollCallback(window, scroll_callback);
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

		while (!glfwWindowShouldClose(window))
		{
			glClearColor(0.23f, 0.11f, 0.32f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT);
			try
			{
				prepare_input(i);
				std::unique_ptr<RayTracer::rgb> ppixels = RayTracer::render(i, proj_type);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, window_width, window_height, 0, GL_RGB, GL_UNSIGNED_BYTE, ppixels.get());
			}
			catch (RayTracer::RayTraceException& e)
			{
				std::cout << "Exception caught :: " << e.what() << std::endl;
				break;
			}
			glUseProgram(shader_program);
			glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
			check_btn_press(window);
			glfwSwapBuffers(window);
			glfwPollEvents();
		}

		glDeleteBuffers(1, &VBO);
		glDeleteVertexArrays(1, &VAO);
		glDeleteProgram(shader_program);
		glfwDestroyWindow(window);
		glfwTerminate();
	}
	if(pmodel) delete_texture(pmodel.get());
	if(init_called) RayTracer::clear();
	if(i.rotator.pmatrix) delete[] i.rotator.pmatrix;
	if(i.translator.pmatrix) delete[] i.translator.pmatrix;
}

void check_btn_press(GLFWwindow* window)
{
	translater.x = 0.0;
	translater.y = 0.0;
	translater.z = 0.0;
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		if (proj_type == RayTracer::Projection::PERSPECTIVE)
		{
			translater.x -= pcamera->front.x;
			translater.y -= pcamera->front.y;
			translater.z -= pcamera->front.z;
		}
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		translater.x += pcamera->right.x;
		translater.y += pcamera->right.y;
		translater.z += pcamera->right.z;
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		if (proj_type == RayTracer::Projection::PERSPECTIVE)
		{
			translater.x += pcamera->front.x;
			translater.y += pcamera->front.y;
			translater.z += pcamera->front.z;
		}
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		translater.x -= pcamera->right.x;
		translater.y -= pcamera->right.y;
		translater.z -= pcamera->right.z;
	}
	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
	{
		lmb_hold = true;
	}
	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE)
	{
		lmb_hold = false;
		first_lmb = true;
	}
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	if (lmb_hold && proj_type == RayTracer::Projection::PERSPECTIVE)
	{
		if (first_lmb)
		{
			last_x = window_width / 2;
			last_y = window_height / 2;
			glfwSetCursorPos(window, last_x, last_y);
			first_lmb = false;
		}
		else
		{
			double xdiff = xpos - last_x;
			last_x = xpos;
			yaw += xdiff;
			double yaw_in_rad = (yaw * 3.141592653589793) / 180.0;

			double ydiff = last_y - ypos;
			last_y = ypos;
			pitch += ydiff;
			if (pitch > 89.0) pitch = 89.0;
			if (pitch < -89.0) pitch = -89.0;

			double pitch_in_rad = (pitch * 3.141592653589793) / 180.0;

			pcamera->right.x = cos(yaw_in_rad);
			pcamera->right.y = 0;
			pcamera->right.z = -sin(yaw_in_rad);

			pcamera->up.x = sin(yaw_in_rad) * sin(pitch_in_rad);
			pcamera->up.y = cos(pitch_in_rad);
			pcamera->up.z = cos(yaw_in_rad) * sin(pitch_in_rad);

			pcamera->front.x = sin(yaw_in_rad) * cos(pitch_in_rad);
			pcamera->front.y = -sin(pitch_in_rad);
			pcamera->front.z = cos(yaw_in_rad) * cos(pitch_in_rad);
		}
	}
}

void scroll_callback(GLFWwindow* window, double xpos, double ypos)
{
	if (ypos > 0.0) if (fov <= 0.0) fov += 1.0; else fov -= 1.0;
	if (ypos < 0.0) if (fov >= 180.0) fov -= 1.0; else fov += 1.0;
}

void prepare_input(RayTracer::input& i)
{
	i.fov = fov;

	i.translator.pmatrix = new double[16];
	i.translator.pmatrix[0] = 1;
	i.translator.pmatrix[1] = 0;
	i.translator.pmatrix[2] = 0;
	i.translator.pmatrix[3] = translater.x;
	i.translator.pmatrix[4] = 0;
	i.translator.pmatrix[5] = 1;
	i.translator.pmatrix[6] = 0;
	i.translator.pmatrix[7] = translater.y;
	i.translator.pmatrix[8] = 0;
	i.translator.pmatrix[9] = 0;
	i.translator.pmatrix[10] = 1;
	i.translator.pmatrix[11] = translater.z;
	i.translator.pmatrix[12] = 0;
	i.translator.pmatrix[13] = 0;
	i.translator.pmatrix[14] = 0;
	i.translator.pmatrix[15] = 1;

	i.rotator.pmatrix = new double[16];
	i.rotator.pmatrix[0] = pcamera->right.x;
	i.rotator.pmatrix[1] = pcamera->up.x;
	i.rotator.pmatrix[2] = pcamera->front.x;
	i.rotator.pmatrix[3] = 0;
	i.rotator.pmatrix[4] = pcamera->right.y;
	i.rotator.pmatrix[5] = pcamera->up.y;
	i.rotator.pmatrix[6] = pcamera->front.y;
	i.rotator.pmatrix[7] = 0;
	i.rotator.pmatrix[8] = pcamera->right.z;
	i.rotator.pmatrix[9] = pcamera->up.z;
	i.rotator.pmatrix[10] = pcamera->front.z;
	i.rotator.pmatrix[11] = 0;
	i.rotator.pmatrix[12] = 0;
	i.rotator.pmatrix[13] = 0;
	i.rotator.pmatrix[14] = 0;
	i.rotator.pmatrix[15] = 1;
}