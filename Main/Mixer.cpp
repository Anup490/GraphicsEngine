#include <iostream>
#include <memory>
#include <vector>
#include "FileReader.h"
#include "RayTracer.h"
#include "Rasterizer.h"
#include <glad/glad.h>
#include <glfw3.h>
#define _USE_MATH_DEFINES
#include <math.h>
#define TO_RADIAN(x) (x * M_PI)/180

double fov = 90.0, last_x = 0.0, last_y = 0.0, yaw_ray = 0.0, pitch_ray = 0.0;
bool lmb_hold = false, first_lmb = true;
int window_width = 1024, window_height = 768;
double near_plane = -1.0, far_plane = -100.0, yaw_ras = 180.0, pitch_ras = 0.0;;

Base::model* p_ray_camera = 0;
Base::model* p_ras_camera = 0;
Engine::Projection proj_type = Engine::Projection::PERSPECTIVE;
Base::vec3 translater;

void check_btn_press(GLFWwindow* window);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xpos, double ypos);
void prepare_raytrace_input(Engine::raytrace_input& i);
void prepare_raster_input(Engine::raster_input& i);

Base::model* prepare_gltf_model_data(Base::model_info info) throw(FileReadException);

void mixer()
{
	std::cout << "Loading..." << std::endl;
	const char* window_title = "GraphicsEngine";
	Engine::raytrace_input ray_i;
	ray_i.translator.pmatrix = new double[16];
	ray_i.rotator.pmatrix = new double[16];
	Engine::RayTracer* praytracer = 0;
	Engine::raster_input ras_i;
	ras_i.view.pmatrix = new double[16];
	ras_i.projection.pmatrix = new double[16];
	Engine::Rasterizer* prasterizer = 0;
	Engine::rgb* prgb = 0;
	try
	{
		std::shared_ptr<std::vector<Base::model*>> praymodels = prepare_data(p_ray_camera);
		std::unique_ptr<Base::cubemap> praycubemap = prepare_cubemap("D:/Projects/C++/3DImporter/Assets/skybox");
		praytracer = new Engine::RayTracer(praymodels, praycubemap.get(), window_width, window_height);
		delete_cubemap(praycubemap);
		delete_data(praymodels);

		std::shared_ptr<std::vector<Base::model*>> prasmodels(new std::vector<Base::model*>);
		Base::model* pmodel = prepare_gltf_model_data({ "D:/Projects/C++/3DImporter/Assets/airplane/scene.gltf", Base::vec3{} });

		Base::model* plight = new Base::model;
		plight->position = Base::vec3{ -75.0, 100.0, -100.0 };
		plight->emissive_color = Base::vec3{ 1.0, 1.0, 1.0 };
		plight->m_type = Base::model_type::LIGHT;

		p_ras_camera = new Base::model;
		p_ras_camera->position = Base::vec3{ 0, 0, 3 };
		p_ras_camera->front = Base::vec3{ 0, 0, -1 };
		p_ras_camera->right = Base::vec3{ -1, 0, 0 };
		p_ras_camera->up = Base::vec3{ 0, 1, 0 };
		p_ras_camera->m_type = Base::model_type::CAMERA;

		prasmodels->push_back(pmodel);
		prasmodels->push_back(plight);
		prasmodels->push_back(p_ras_camera);

		std::unique_ptr<Base::cubemap> prascubemap = prepare_cubemap("D:/Projects/C++/3DImporter/Assets/skybox");
		prasterizer = new Engine::Rasterizer(prasmodels, prascubemap.get(), window_width, window_height);
		delete_cubemap(prascubemap);
		delete_data(prasmodels);
	}
	catch (std::exception& e)
	{
		std::cout << "Exception thrown :: " << e.what() << std::endl;
	}
	if (praytracer && prasterizer)
	{
		std::cout << "Opening window" << std::endl;
		const char* window_title = "GraphicsEngine";

		GLfloat square_vertices[] =
		{
			-1.0f, -1.0f, 0.0f,   0.0f, 1.0f,
			 1.0f, -1.0f, 0.0f,   1.0f, 1.0f,
			-1.0f,  1.0f, 0.0f,   0.0f, 0.0f,
			 1.0f,  1.0f, 0.0f,   1.0f, 0.0f
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

		prgb = new Engine::rgb[window_width * window_height];

		while (!glfwWindowShouldClose(window))
		{
			glClearColor(0.23f, 0.11f, 0.32f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT);
			try
			{
				prepare_raytrace_input(ray_i);
				prepare_raster_input(ras_i);
				Engine::pixels raytrace_pixels = praytracer->render(ray_i, proj_type);
				Engine::pixels raster_pixels = prasterizer->render(ras_i, p_ras_camera);
				Engine::mix(raster_pixels, raytrace_pixels, prgb);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, window_width, window_height, 0, GL_RGB, GL_UNSIGNED_BYTE, prgb);
			}
			catch (Engine::RayTraceException& e)
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
	if (praytracer) delete praytracer;
	if (ray_i.rotator.pmatrix) delete[] ray_i.rotator.pmatrix;
	if (ray_i.translator.pmatrix) delete[] ray_i.translator.pmatrix;
	if (p_ray_camera) delete p_ray_camera;
	if (p_ras_camera) delete p_ras_camera;
	if (prgb) delete[] prgb;
}

void check_btn_press(GLFWwindow* window)
{
	translater.x = 0.0;
	translater.y = 0.0;
	translater.z = 0.0;
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		if (proj_type == Engine::Projection::PERSPECTIVE)
		{
			translater.x -= p_ray_camera->front.x;
			translater.y -= p_ray_camera->front.y;
			translater.z -= p_ray_camera->front.z;
		}
		p_ras_camera->position.x += p_ras_camera->front.x;
		p_ras_camera->position.y += p_ras_camera->front.y;
		p_ras_camera->position.z += p_ras_camera->front.z;
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		translater.x -= p_ray_camera->right.x;
		translater.y -= p_ray_camera->right.y;
		translater.z -= p_ray_camera->right.z;

		p_ras_camera->position.x -= p_ras_camera->right.x;
		p_ras_camera->position.y -= p_ras_camera->right.y;
		p_ras_camera->position.z -= p_ras_camera->right.z;
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		if (proj_type == Engine::Projection::PERSPECTIVE)
		{
			translater.x += p_ray_camera->front.x;
			translater.y += p_ray_camera->front.y;
			translater.z += p_ray_camera->front.z;
		}

		p_ras_camera->position.x -= p_ras_camera->front.x;
		p_ras_camera->position.y -= p_ras_camera->front.y;
		p_ras_camera->position.z -= p_ras_camera->front.z;
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		translater.x += p_ray_camera->right.x;
		translater.y += p_ray_camera->right.y;
		translater.z += p_ray_camera->right.z;

		p_ras_camera->position.x += p_ras_camera->right.x;
		p_ras_camera->position.y += p_ras_camera->right.y;
		p_ras_camera->position.z += p_ras_camera->right.z;
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
	if (lmb_hold && proj_type == Engine::Projection::PERSPECTIVE)
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
			yaw_ray += xdiff;
			yaw_ras += xdiff;
			double yaw_ray_in_rad = (yaw_ray * M_PI) / 180.0;
			double yaw_ras_in_rad = (yaw_ras * M_PI) / 180.0;

			double y_ray_diff = last_y - ypos;
			double y_ras_diff = ypos - last_y;
			last_y = ypos;
			pitch_ray += y_ray_diff;
			pitch_ras += y_ras_diff;
			if (pitch_ray > 89.0) pitch_ray = 89.0;
			if (pitch_ray < -89.0) pitch_ray = -89.0;
			if (pitch_ras > 89.0) pitch_ras = 89.0;
			if (pitch_ras < -89.0) pitch_ras = -89.0;

			double pitch_ray_in_rad = (pitch_ray * M_PI) / 180.0;
			double pitch_ras_in_rad = (pitch_ras * M_PI) / 180.0;

			p_ray_camera->right.x = cos(yaw_ray_in_rad);
			p_ray_camera->right.y = 0;
			p_ray_camera->right.z = sin(yaw_ray_in_rad);

			p_ray_camera->up.x = -sin(yaw_ray_in_rad) * sin(pitch_ray_in_rad);
			p_ray_camera->up.y = cos(pitch_ray_in_rad);
			p_ray_camera->up.z = cos(yaw_ray_in_rad) * sin(pitch_ray_in_rad);

			p_ray_camera->front.x = -sin(yaw_ray_in_rad) * cos(pitch_ray_in_rad);
			p_ray_camera->front.y = -sin(pitch_ray_in_rad);
			p_ray_camera->front.z = cos(yaw_ray_in_rad) * cos(pitch_ray_in_rad);

			p_ras_camera->right.x = cos(yaw_ras_in_rad);
			p_ras_camera->right.y = 0;
			p_ras_camera->right.z = -sin(yaw_ras_in_rad);

			p_ras_camera->up.x = sin(yaw_ras_in_rad) * sin(pitch_ras_in_rad);
			p_ras_camera->up.y = cos(pitch_ras_in_rad);
			p_ras_camera->up.z = cos(yaw_ras_in_rad) * sin(pitch_ras_in_rad);

			p_ras_camera->front.x = sin(yaw_ras_in_rad) * cos(pitch_ras_in_rad);
			p_ras_camera->front.y = -sin(pitch_ras_in_rad);
			p_ras_camera->front.z = cos(yaw_ras_in_rad) * cos(pitch_ras_in_rad);
		}
	}
}

void scroll_callback(GLFWwindow* window, double xpos, double ypos)
{
	if (ypos > 0.0) if (fov <= 0.0) fov += 1.0; else fov -= 1.0;
	if (ypos < 0.0) if (fov >= 180.0) fov -= 1.0; else fov += 1.0;
}

void prepare_raytrace_input(Engine::raytrace_input& i)
{
	i.fov = fov;
	i.near = (proj_type == Engine::Projection::PERSPECTIVE) ? 1.0 : 20.0;
	i.far = i.near + 100.0;
	i.proj_type = proj_type;

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

	i.rotator.pmatrix[0] = p_ray_camera->right.x;
	i.rotator.pmatrix[1] = p_ray_camera->up.x;
	i.rotator.pmatrix[2] = p_ray_camera->front.x;
	i.rotator.pmatrix[3] = 0;
	i.rotator.pmatrix[4] = p_ray_camera->right.y;
	i.rotator.pmatrix[5] = p_ray_camera->up.y;
	i.rotator.pmatrix[6] = p_ray_camera->front.y;
	i.rotator.pmatrix[7] = 0;
	i.rotator.pmatrix[8] = p_ray_camera->right.z;
	i.rotator.pmatrix[9] = p_ray_camera->up.z;
	i.rotator.pmatrix[10] = p_ray_camera->front.z;
	i.rotator.pmatrix[11] = 0;
	i.rotator.pmatrix[12] = 0;
	i.rotator.pmatrix[13] = 0;
	i.rotator.pmatrix[14] = 0;
	i.rotator.pmatrix[15] = 1;
}

void prepare_raster_input(Engine::raster_input& i)
{
	i.view.pmatrix[0] = p_ras_camera->right.x;
	i.view.pmatrix[1] = p_ras_camera->right.y;
	i.view.pmatrix[2] = p_ras_camera->right.z;
	i.view.pmatrix[3] = -(p_ras_camera->position.x * p_ras_camera->right.x + p_ras_camera->position.y * p_ras_camera->right.y + p_ras_camera->position.z * p_ras_camera->right.z);
	i.view.pmatrix[4] = p_ras_camera->up.x;
	i.view.pmatrix[5] = p_ras_camera->up.y;
	i.view.pmatrix[6] = p_ras_camera->up.z;
	i.view.pmatrix[7] = -(p_ras_camera->position.x * p_ras_camera->up.x + p_ras_camera->position.y * p_ras_camera->up.y + p_ras_camera->position.z * p_ras_camera->up.z);
	i.view.pmatrix[8] = p_ras_camera->front.x;
	i.view.pmatrix[9] = p_ras_camera->front.y;
	i.view.pmatrix[10] = p_ras_camera->front.z;
	i.view.pmatrix[11] = -(p_ras_camera->position.x * p_ras_camera->front.x + p_ras_camera->position.y * p_ras_camera->front.y + p_ras_camera->position.z * p_ras_camera->front.z);
	i.view.pmatrix[12] = 0;
	i.view.pmatrix[13] = 0;
	i.view.pmatrix[14] = 0;
	i.view.pmatrix[15] = 1;

	double n = near_plane;
	double f = far_plane;
	double l = n * tan(TO_RADIAN(fov / 2.0));
	double r = -n * tan(TO_RADIAN(fov / 2.0));
	double b = n * tan(TO_RADIAN(fov / 2.0));
	double t = -n * tan(TO_RADIAN(fov / 2.0));

	i.projection.pmatrix[0] = (2 * n) / (r - l);
	i.projection.pmatrix[1] = 0;
	i.projection.pmatrix[2] = (r + l) / (r - l);
	i.projection.pmatrix[3] = 0;
	i.projection.pmatrix[4] = 0;
	i.projection.pmatrix[5] = (2 * n) / (t - b);
	i.projection.pmatrix[6] = (t + b) / (t - b);
	i.projection.pmatrix[7] = 0;
	i.projection.pmatrix[8] = 0;
	i.projection.pmatrix[9] = 0;
	i.projection.pmatrix[10] = (-f - n) / (f - n);
	i.projection.pmatrix[11] = (-2 * f * n) / (f - n);
	i.projection.pmatrix[12] = 0;
	i.projection.pmatrix[13] = 0;
	i.projection.pmatrix[14] = -1;
	i.projection.pmatrix[15] = 0;
}
	
