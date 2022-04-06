#pragma once
#include "Core.h"

#include <exception>

namespace std
{
	template <class _Ty, class _Dx>
	class unique_ptr;
}

struct FileReadException : std::exception
{
	std::string message;
	FileReadException(std::string message) : message(message) {}
	char const* what() const override { return message.c_str(); }
};

std::unique_ptr<Core::model> prepare_gltf_model_data(Core::model_info info) throw(FileReadException);
std::unique_ptr<Core::model> prepare_spheres();
std::unique_ptr<Core::model> prepare_boxes();
std::unique_ptr<Core::cubemap> prepare_cubemap(const char* file_path);
Core::texture get_texture(const char* file_path);
void delete_texture(Core::model* pmodel);
std::string extract_file(const char* path);