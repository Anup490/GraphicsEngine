#pragma once

namespace RayTracer
{
	namespace Texture
	{
		RUN_ON_GPU
		Core::vec3 get_color(const Core::vec3& texcoord, const Core::texture& tex)
		{
			unsigned x = texcoord.x * tex.width;
			unsigned y = (1 - texcoord.y) * tex.height;
			unsigned index = (y * tex.width + x) * tex.channels;
			if (tex.width == 0) return Core::vec3{};
			double r = double(tex.ptextures[index]) / 255.0;
			double g = double(tex.ptextures[index + 1]) / 255.0;
			double b = double(tex.ptextures[index + 2]) / 255.0;
			return Core::vec3{ r, g, b };
		}
	}
}