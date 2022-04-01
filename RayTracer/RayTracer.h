#pragma once
#include "PublicBase.h"
#include <exception>

namespace std
{
	template <class _Ty>
	struct default_delete;

	template <class _Ty>
	class shared_ptr;

	template <class _Ty, class _Dx = std::default_delete<_Ty>>
	class unique_ptr;
}

namespace RayTracer
{
	class RayTraceException : std::exception {
		const char* message;
	public:
		RayTraceException(const char* message) : message(message) {}
		char const* what() const override { return message; }
	};
	void init(std::shared_ptr<std::vector<Core::model*>> pmodels, Core::cubemap* pcubemap, int width, int height) throw(RayTraceException);
	std::unique_ptr<rgb> render(input i, Projection proj_type) throw(RayTraceException);
	void clear();
}