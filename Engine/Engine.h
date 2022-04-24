#pragma once
#include "Base.h"

namespace std
{
	template <class _Ty>
	struct default_delete;

	template <class _Ty>
	class shared_ptr;

	template <class _Ty, class _Dx = std::default_delete<_Ty>>
	class unique_ptr;
}

namespace Engine
{
	struct rgb
	{
		unsigned char r = 0, g = 0, b = 0;
	};

	struct pixels
	{
		rgb* data = 0;
		int width = 0, height = 0;
		pixels(int width, int height) : width(width), height(height) {}
	};

	enum class Projection { PERSPECTIVE, ORTHOGRAPHIC };

	struct raytrace_input
	{
		double fov = 90.0;
		double near = 0.0;
		double far = 0.0;
		Projection proj_type;
		Base::mat4 translator;
		Base::mat4 rotator;
	};
	
	struct raster_input
	{
		Base::mat4 view;
		Base::mat4 projection;
	};
	
}