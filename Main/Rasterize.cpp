#include <iostream>
#include <memory>
#include <vector>
#include "FileReader.h"
#include "Rasterizer.h"
#include <glad/glad.h>
#include <glfw3.h>

#define _USE_MATH_DEFINES
#include <math.h>
#define TO_RADIAN(x) (x * M_PI)/180

Base::model* prepare_gltf_model_data(Base::model_info info) throw(FileReadException);

namespace Rasterize
{
	int window_width = 1024, window_height = 768;
	double fov = 90.0, near_plane = -1.0, far_plane = -100.0, last_x = 0.0, last_y = 0.0, yaw = 180.0, pitch = 0.0;
	bool lmb_hold = false, first_lmb = true;
	Base::model* pcamera = 0;

	void check_btn_press(GLFWwindow* window);
	void mouse_callback(GLFWwindow* window, double xpos, double ypos);
	void scroll_callback(GLFWwindow* window, double xpos, double ypos);
	void prepare_raster_input(Engine::raster_input& i);

	void rasterize()
	{
		std::cout << "Loading..." << std::endl;
		const char* window_title = "GraphicsEngine";
		Engine::raster_input i;
		i.view.pmatrix = new double[16];
		i.projection.pmatrix = new double[16];
		Engine::Rasterizer* prasterizer = 0;
		try
		{
			std::shared_ptr<std::vector<Base::model*>> pmodels(new std::vector<Base::model*>);
			Base::model* pmodel = prepare_gltf_model_data({ "D:/Projects/C++/3DImporter/Assets/airplane/scene.gltf", Base::vec3{} });

			Base::model* plight = new Base::model;
			plight->position = Base::vec3{ 75.0, 100.0, -100.0 };
			plight->emissive_color = Base::vec3{ 1.0, 1.0, 1.0 };
			plight->m_type = Base::model_type::LIGHT;

			pcamera = new Base::model;
			pcamera->position = Base::vec3{ 0, 0, 3 };
			pcamera->front = Base::vec3{ 0, 0, -1 };
			pcamera->right = Base::vec3{ -1, 0, 0 };
			pcamera->up = Base::vec3{ 0, 1, 0 };
			pcamera->m_type = Base::model_type::CAMERA;
			
			pmodels->push_back(pmodel);
			pmodels->push_back(plight);
			pmodels->push_back(pcamera);

			std::unique_ptr<Base::cubemap> pcubemap = prepare_cubemap("D:/Projects/C++/3DImporter/Assets/skybox");
			prasterizer = new Engine::Rasterizer(pmodels, pcubemap.get(), window_width, window_height);
			delete_cubemap(pcubemap);
			delete_data(pmodels);
		}
		catch (std::exception& e)
		{
			std::cout << "Exception thrown :: " << e.what() << std::endl;
		}
		if (prasterizer)
		{
			std::cout << "Opening window" << std::endl;
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
					prepare_raster_input(i);
					Engine::pixels ppixels = prasterizer->render(i, pcamera);
					glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, window_width, window_height, 0, GL_RGB, GL_UNSIGNED_BYTE, ppixels.data);
				}
				catch (Engine::RasterizeException& e)
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
		if (prasterizer) delete prasterizer;
		if (i.view.pmatrix) delete[] i.view.pmatrix;
		if (i.projection.pmatrix) delete[] i.projection.pmatrix;
		if (pcamera) delete pcamera;
	}

	void check_btn_press(GLFWwindow* window)
	{
		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		{
			pcamera->position.x += pcamera->front.x;
			pcamera->position.y += pcamera->front.y;
			pcamera->position.z += pcamera->front.z;
		}
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		{
			pcamera->position.x += pcamera->right.x;
			pcamera->position.y += pcamera->right.y;
			pcamera->position.z += pcamera->right.z;
		}
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		{
			pcamera->position.x -= pcamera->front.x;
			pcamera->position.y -= pcamera->front.y;
			pcamera->position.z -= pcamera->front.z;
		}
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		{
			pcamera->position.x -= pcamera->right.x;
			pcamera->position.y -= pcamera->right.y;
			pcamera->position.z -= pcamera->right.z;
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
		if (lmb_hold)
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
				double yaw_in_rad = (yaw * M_PI) / 180.0;
				double ydiff = ypos - last_y;
				last_y = ypos;
				pitch += ydiff;
				if (pitch > 89.0) pitch = 89.0;
				if (pitch < -89.0) pitch = -89.0;

				double pitch_in_rad = (pitch * M_PI) / 180.0;

				pcamera->right.x = cos(yaw_in_rad);
				pcamera->right.y = 0;
				pcamera->right.z = sin(yaw_in_rad);

				pcamera->up.x = -sin(yaw_in_rad) * sin(pitch_in_rad);
				pcamera->up.y = cos(pitch_in_rad);
				pcamera->up.z = cos(yaw_in_rad) * sin(pitch_in_rad);

				pcamera->front.x = -sin(yaw_in_rad) * cos(pitch_in_rad);
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

	void prepare_raster_input(Engine::raster_input& i)
	{
		i.view.pmatrix[0] = pcamera->right.x;
		i.view.pmatrix[1] = pcamera->right.y;
		i.view.pmatrix[2] = pcamera->right.z;
		i.view.pmatrix[3] = -(pcamera->position.x * pcamera->right.x + pcamera->position.y * pcamera->right.y + pcamera->position.z * pcamera->right.z);
		i.view.pmatrix[4] = pcamera->up.x;
		i.view.pmatrix[5] = pcamera->up.y;
		i.view.pmatrix[6] = pcamera->up.z;
		i.view.pmatrix[7] = -(pcamera->position.x * pcamera->up.x + pcamera->position.y * pcamera->up.y + pcamera->position.z * pcamera->up.z);
		i.view.pmatrix[8] = pcamera->front.x;
		i.view.pmatrix[9] = pcamera->front.y;
		i.view.pmatrix[10] = pcamera->front.z;
		i.view.pmatrix[11] = -(pcamera->position.x * pcamera->front.x + pcamera->position.y * pcamera->front.y + pcamera->position.z * pcamera->front.z);
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
		i.projection.pmatrix[10] = (-f - n)/(f - n);
		i.projection.pmatrix[11] = (-2 * f * n) / (f - n);
		i.projection.pmatrix[12] = 0;
		i.projection.pmatrix[13] = 0;
		i.projection.pmatrix[14] = -1;
		i.projection.pmatrix[15] = 0;
	}
}