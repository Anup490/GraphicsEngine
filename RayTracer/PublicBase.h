#pragma once
#include "Core.h"

namespace RayTracer
{
	struct pixels
	{
		Core::vec3* data = 0;
		int width = 0, height = 0;
		pixels(int width, int height) : width(width), height(height) {}
	};

	enum class Projection { PERSPECTIVE, ORTHOGRAPHIC };
}