#pragma once
#include "Engine.h"
#include <exception>

namespace Engine
{
	struct RayTraceException : std::exception {
		const char* message;
		RayTraceException(const char* message) : message(message) {}
		char const* what() const override { return message; }
	};

	class RayTracer
	{
		struct RayTracerCore* pcore;
	public:
		RayTracer(std::shared_ptr<std::vector<Base::model*>> pmodels, Base::cubemap* pcubemap, int width, int height) throw(RayTraceException);
		pixels render(const raytrace_input& i, Projection proj_type) throw(RayTraceException);
		~RayTracer();
	};
}