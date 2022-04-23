#pragma once

namespace Engine
{
	namespace Texture
	{
		RUN_ON_GPU
		Base::vec3 get_color(const Base::vec3& texcoord, const Base::texture& tex)
		{
			if (tex.width == 0) return Base::vec3{};
			unsigned x = texcoord.x * tex.width;
			unsigned y = (1 - texcoord.y) * tex.height;
			unsigned index = (y * tex.width + x) * tex.channels;
			if (index < 0.0) index = 0.0;
			if (index > (tex.width * tex.height * tex.channels)) 
				index = (tex.width * tex.height - 1) * tex.channels;
			double r = double(tex.ptextures[index]) / 255.0;
			double g = double(tex.ptextures[index + 1]) / 255.0;
			double b = double(tex.ptextures[index + 2]) / 255.0;
			return Base::vec3{ r, g, b };
		}
	}
}