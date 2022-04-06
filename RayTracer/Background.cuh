#include "PrivateBase.cuh"
#include "Texture.cuh"

namespace RayTracer
{
	RUN_ON_GPU
	Core::vec3 get_background_color(const Core::cubemap* dcubemap, const Core::vec3& dir)
	{
		Core::box unit{ Core::vec3{ -1.0, -1.0, -1.0}, Core::vec3{ 1.0, 1.0, 1.0 } };
		double dist;
		Box::does_intersect(unit, ray{ Core::vec3{}, dir }, dist);
		Core::vec3 phit = dir * dist;
		face f = Box::get_face(&unit, phit);
		if (f == face::LEFT)
		{
			double z = (phit.z + 1.0) / 2.0;
			double y = (1.0 - phit.y) / 2.0;
			return Texture::get_color(Core::vec3{ z, y }, dcubemap->left);
		}
		if (f == face::RIGHT)
		{
			double z = (phit.z + 1.0) / 2.0;
			double y = (1.0 - phit.y) / 2.0;
			z = 1 - z;
			return Texture::get_color(Core::vec3{ z, y }, dcubemap->right);
		}
		if (f == face::BOTTOM)
		{
			double x = (phit.x + 1.0) / 2.0;
			double z = (1.0 - phit.z) / 2.0;
			x = 1 - x;
			return Texture::get_color(Core::vec3{ x, z }, dcubemap->bottom);
		}
		if (f == face::TOP)
		{
			double x = (phit.x + 1.0) / 2.0;
			double z = (1.0 - phit.z) / 2.0;
			x = 1 - x;
			z = 1 - z;
			return Texture::get_color(Core::vec3{ x, z }, dcubemap->top);
		}
		if (f == face::FRONT)
		{
			double x = (phit.x + 1.0) / 2.0;
			double y = (1.0 - phit.y) / 2.0;
			x = 1 - x;
			return Texture::get_color(Core::vec3{ x, y }, dcubemap->front);
		}
		if (f == face::BACK)
		{
			double x = (phit.x + 1.0) / 2.0;
			double y = (1.0 - phit.y) / 2.0;
			return Texture::get_color(Core::vec3{ x, y }, dcubemap->back);
		}
		return Core::vec3{ 1.0, 1.0, 1.0 };
	}

	RUN_ON_GPU
	void to_cubemap(const Core::texture& tex, Core::cubemap& cubemap)
	{
		cubemap.left.ptextures = tex.ptextures;
		cubemap.left.channels = tex.channels;
		cubemap.left.width = tex.width;
		cubemap.left.height = tex.height;

		cubemap.right.ptextures = tex.ptextures;
		cubemap.right.channels = tex.channels;
		cubemap.right.width = tex.width;
		cubemap.right.height = tex.height;

		cubemap.bottom.ptextures = tex.ptextures;
		cubemap.bottom.channels = tex.channels;
		cubemap.bottom.width = tex.width;
		cubemap.bottom.height = tex.height;

		cubemap.top.ptextures = tex.ptextures;
		cubemap.top.channels = tex.channels;
		cubemap.top.width = tex.width;
		cubemap.top.height = tex.height;

		cubemap.front.ptextures = tex.ptextures;
		cubemap.front.channels = tex.channels;
		cubemap.front.width = tex.width;
		cubemap.front.height = tex.height;

		cubemap.back.ptextures = tex.ptextures;
		cubemap.back.channels = tex.channels;
		cubemap.back.width = tex.width;
		cubemap.back.height = tex.height;
	}

}