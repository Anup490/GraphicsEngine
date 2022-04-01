#include "PrivateBase.cuh"

namespace RayTracer
{
	enum class face { LEFT, RIGHT, BOTTOM, TOP, FRONT, BACK, NONE };

	RUN_ON_GPU
	face get_face(const Core::box& box, const Core::vec3& phit)
	{
		if (box.min.x >= phit.x) return face::LEFT;
		else if (box.max.x <= phit.x) return face::RIGHT;
		else if (box.min.y >= phit.y) return face::BOTTOM;
		else if (box.max.y <= phit.y) return face::TOP;
		else if (box.min.z >= phit.z) return face::FRONT;
		else if (box.max.z <= phit.z) return face::BACK;
		return face::NONE;
	}

	RUN_ON_GPU
	Core::vec3 get_background_color(const Core::cubemap* dcubemap, const Core::vec3& dir)
	{
		Core::box unit{ Core::vec3{ -1.0, -1.0, -1.0}, Core::vec3{ 1.0, 1.0, 1.0 } };
		double dist;
		Box::does_intersect(unit, ray{ Core::vec3{}, dir }, dist);
		Core::vec3 phit = dir * dist;
		face f = get_face(unit, phit);
		Core::vec3 color;
		if (f == face::LEFT)
		{
			double z = (phit.z + 1.0) / 2.0;
			double y = (1.0 - phit.y) / 2.0;
			z = 1 - z;
			color = Texture::get_color(Core::vec3{ z, y }, dcubemap->left);
		}
		else if (f == face::RIGHT)
		{
			double z = (phit.z + 1.0) / 2.0;
			double y = (1.0 - phit.y) / 2.0;
			color = Texture::get_color(Core::vec3{ z, y }, dcubemap->right);
		}
		else if (f == face::BOTTOM)
		{
			double x = (phit.x + 1.0) / 2.0;
			double z = (1.0 - phit.z) / 2.0;
			z = 1 - z;
			color = Texture::get_color(Core::vec3{ x, z }, dcubemap->bottom);
		}
		else if (f == face::TOP)
		{
			double x = (phit.x + 1.0) / 2.0;
			double z = (1.0 - phit.z) / 2.0;
			color = Texture::get_color(Core::vec3{ x, z }, dcubemap->top);
		}
		else if (f == face::FRONT)
		{
			double x = (phit.x + 1.0) / 2.0;
			double y = (1.0 - phit.y) / 2.0;
			color = Texture::get_color(Core::vec3{ x, y }, dcubemap->front);
		}
		else
		{
			double x = (phit.x + 1.0) / 2.0;
			double y = (1.0 - phit.y) / 2.0;
			x = 1 - x;
			color = Texture::get_color(Core::vec3{ x, y }, dcubemap->back);
		}
		return get_clamped(color);
	}
}