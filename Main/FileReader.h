#pragma once
#include "Core.h"

#include <exception>

namespace std
{
	template <class _Ty, class _Dx>
	class unique_ptr;

	template <class _Ty>
	class shared_ptr;

	template <class _Ty>
	class allocator;

	template <class _Ty, class _Alloc = allocator<_Ty>>
	class vector;
}

struct FileReadException : std::exception
{
	std::string message;
	FileReadException(std::string message) : message(message) {}
	char const* what() const override { return message.c_str(); }
};

std::shared_ptr<std::vector<Core::model*>> prepare_data(Core::model*& pcamera);
void delete_data(std::shared_ptr<std::vector<Core::model*>> pmodels);
std::unique_ptr<Core::cubemap> prepare_cubemap(const char* file_path);
std::string extract_file(const char* path);
void delete_cubemap(std::unique_ptr<Core::cubemap>& pcubemap);