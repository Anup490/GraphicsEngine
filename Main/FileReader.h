#pragma once
#include "Core.h"
#include <exception>

namespace std
{
	template <class _Ty>
	class shared_ptr;
}

struct FileReadException : std::exception
{
	const char* message;
	FileReadException(const char* message) : message(message) {}
	char const* what() const override { return message; }
};

std::shared_ptr<Core::model> prepare_gltf_model_data(const char* file_path) throw(FileReadException);