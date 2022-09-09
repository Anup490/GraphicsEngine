#pragma once
#include "Base.h"
#include <memory>

namespace Engine
{
	struct rgb
	{
		unsigned char r = 0, g = 0, b = 0;
	};

	struct pixels
	{
		rgb* data = 0;
		double* depth = 0;
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
	
	void mix(const pixels& pixels1, const pixels& pixels2, rgb* prgb);
}