#pragma once
#include "Triangle.cuh"

namespace RayTracer
{
	namespace Texture
	{
		RUN_ON_GPU
		Core::vec3 get_color(const Core::vec3& texcoord, const texture& tex)
		{
			unsigned x = texcoord.x * tex.width;
			unsigned y = texcoord.y * tex.height;
			unsigned index = (y * tex.width + x) * 3.0;
			if (tex.width == 0) return Core::vec3{};
			double r = double(tex.dtextures[index]) / 255.0;
			double g = double(tex.dtextures[index + 1]) / 255.0;
			double b = double(tex.dtextures[index + 2]) / 255.0;
			return Core::vec3{ r, g, b };
		}
	}
}