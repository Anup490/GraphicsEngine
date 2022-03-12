#include "Math.cuh"
#include "Vector.cuh"
#include "Texture.cuh"

namespace RayTracer
{
	RUN_ON_GPU
	Core::vec3 get_texcoord(const void* hit_shape, const Core::vec3& p, Core::shape_type shape)
	{
		if (shape == Core::shape_type::TRIANGLE)
		{
			triangle* ptriangle = (triangle*)hit_shape;
			return Triangle::get_texcoord(*ptriangle, p);
		}
		return Core::vec3{};
	}

	RUN_ON_GPU
	Core::vec3 get_color(const hit& hit_item, ray& rray, const texture& tex)
	{
		if (!tex.dtextures)
		{
			return hit_item.pmodel->surface_color;
		}
		rray.texcoord = get_texcoord(hit_item.shape, rray.phit, hit_item.pmodel->shape);
		return hit_item.pmodel->surface_color * Texture::get_color(rray.texcoord, tex);
	}

	RUN_ON_GPU
	bool detect_hit(const models& models, ray& ray, hit& hit_item)
	{
		double t0 = INFINITY, tnear = INFINITY;
		bool hit = false;
		for (unsigned m = 0; m < models.size; m++)
		{	
			bool has_hit = false;
			if (models.models[m].shape == Core::shape_type::TRIANGLE)
			{
				has_hit = Triangle::detect_hit(models.models[m], ray, hit_item, tnear/*, hit*/);
			}
			if (has_hit) hit = has_hit;
		}
		if (!hit) return hit;
		ray.phit = ray.origin + (ray.dir * tnear);
		triangle* ptriangle = (triangle*)hit_item.shape;
		ray.nhit = ptriangle->normal;
		normalize(ray.nhit);
		return hit;
	}

	RUN_ON_GPU
	double get_glow_val(const model& model, const ray& shadow_ray, double& t0/*, double& glow*/)
	{
		double glow = 1.0;
		if (model.shape == Core::shape_type::TRIANGLE)
		{
			triangle* triangles = (triangle*)model.dshapes;
			for (unsigned i = 0; i < model.shapes_size; i++)
			{
				if (Triangle::does_intersect(triangles[i], shadow_ray, t0))
				{
					glow = 0.0;
					break;
				}
			}
		}
		return glow;
	}
}