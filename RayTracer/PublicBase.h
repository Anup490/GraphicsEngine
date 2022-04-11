#pragma once
#include "Core.h"

namespace RayTracer
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

	struct input
	{
		void* dworld;
		double fov = 90.0;
		Core::mat4 translator;
		Core::mat4 rotator;
	};

	enum class Projection { PERSPECTIVE, ORTHOGRAPHIC };
}