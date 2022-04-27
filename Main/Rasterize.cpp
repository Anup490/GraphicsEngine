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
	double fov = 90.0;
	double near_plane = -1.0;
	double far_plane = -10.0;

	void prepare_raster_input(Engine::raster_input& i);

	void rasterize()
	{
		std::cout << "Loading..." << std::endl;
		const char* window_title = "GraphicsEngine";
		Engine::raster_input i;
		Engine::Rasterizer* prasterizer = 0;
		try
		{
			std::shared_ptr<std::vector<Base::model*>> pmodels(new std::vector<Base::model*>);
			Base::model* pmodel = prepare_gltf_model_data({ "D:/Projects/C++/3DImporter/Assets/airplane/scene.gltf", Base::vec3{} });
			pmodels->push_back(pmodel);
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
			/*glfwSetCursorPosCallback(window, mouse_callback);
			glfwSetScrollCallback(window, scroll_callback);*/
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

			while (!glfwWindowShouldClose(window))
			{
				glClearColor(0.23f, 0.11f, 0.32f, 1.0f);
				glClear(GL_COLOR_BUFFER_BIT);
				try
				{
					prepare_raster_input(i);
					std::unique_ptr<Engine::rgb> ppixels = prasterizer->render(i);
					glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, window_width, window_height, 0, GL_RGB, GL_UNSIGNED_BYTE, ppixels.get());
				}
				catch (Engine::RasterizeException& e)
				{
					std::cout << "Exception caught :: " << e.what() << std::endl;
					break;
				}
				glUseProgram(shader_program);
				glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
				//check_btn_press(window);
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
	}

	void prepare_raster_input(Engine::raster_input& i)
	{
		i.view.pmatrix = new double[16];
		i.view.pmatrix[0] = 1;
		i.view.pmatrix[1] = 0;
		i.view.pmatrix[2] = 0;
		i.view.pmatrix[3] = 0;
		i.view.pmatrix[4] = 0;
		i.view.pmatrix[5] = 1;
		i.view.pmatrix[6] = 0;
		i.view.pmatrix[7] = 0;
		i.view.pmatrix[8] = 0;
		i.view.pmatrix[9] = 0;
		i.view.pmatrix[10] = -1;
		i.view.pmatrix[11] = 3;
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

		i.projection.pmatrix = new double[16];
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