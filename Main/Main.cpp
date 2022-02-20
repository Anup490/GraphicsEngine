#include <iostream>
#include <memory>
#include <vector>
#include "FileReader.h"
#include "RayTracer.h"
#include <glad/glad.h>
#include <glfw3.h>

void check_btn_press(GLFWwindow* window);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xpos, double ypos);

void main()
{
	std::cout << "Loading..." << std::endl;
	int window_width = 640, window_height = 480;
	const char* window_title = "GraphicsEngine";
	bool init_called = false;
	std::unique_ptr<Core::model> pmodel;
	try
	{
		pmodel = prepare_gltf_model_data("D:/Projects/C++/3DImporter/Assets/airplane/scene.gltf");
		Core::model light_model{ Core::vec3{}, Core::vec3{1.0, 1.0, 1.0} };
		std::shared_ptr<std::vector<Core::model*>> pmodels(new std::vector<Core::model*>);
		pmodels->push_back(pmodel.get());
		pmodels->push_back(&light_model);
		RayTracer::init(pmodels, 640, 480);
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
				std::unique_ptr<RayTracer::rgb> ppixels = RayTracer::render(90.0, RayTracer::Projection::PERSPECTIVE);
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
	delete_texture(pmodel.get());
	if(init_called) RayTracer::clear();
}

void check_btn_press(GLFWwindow* window)
{
	if(glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		std::cout << "Presses button W" << std::endl;
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		std::cout << "Presses button A" << std::endl;
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		std::cout << "Presses button S" << std::endl;
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		std::cout << "Presses button D" << std::endl;
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	std::cout << "Mouse movement detected" << std::endl;
}

void scroll_callback(GLFWwindow* window, double xpos, double ypos)
{
	if (ypos > 0.0)
		std::cout << "Zoom in" << std::endl;
	if (ypos < 0.0)
		std::cout << "Zoom out" << std::endl;
}