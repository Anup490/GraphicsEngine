#include "Maths.cuh"
#include "Vector.cuh"
#include "Triangle.cuh"
#include "Sphere.cuh"
#include "Box.cuh"
#include "Background.cuh"

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
	Core::vec3 get_color(const hit& hit_item, const Core::texture& tex, const ray& rray)
	{
		if (!tex.ptextures) return hit_item.pmodel->surface_color;
		if (hit_item.pmodel->s_type == Core::shape_type::TRIANGLE)
		{
			Core::vec3 texcoord = get_texcoord(hit_item.shape, rray.phit, hit_item.pmodel->s_type);
			return hit_item.pmodel->surface_color * Texture::get_color(texcoord, tex);
		}
		Core::cubemap texture_3d;
		to_cubemap(tex, texture_3d);
		if (hit_item.pmodel->s_type == Core::shape_type::SPHERE)
		{
			return hit_item.pmodel->surface_color * get_background_color(&texture_3d, rray.nhit);
		}
		if (hit_item.pmodel->s_type == Core::shape_type::BOX)
		{
			Core::box* pbox = (Core::box*)hit_item.shape;
			Core::vec3 dir = rray.phit - pbox->center;
			return hit_item.pmodel->surface_color * get_background_color(&texture_3d, dir);
		}
		return Core::vec3{};
	}

	RUN_ON_GPU
	Core::vec3 get_specular_val(const hit& hit_item, const Core::texture& tex, const ray& rray)
	{
		if (!tex.ptextures) return Core::vec3{ hit_item.pmodel->smoothness,hit_item.pmodel->smoothness,hit_item.pmodel->smoothness };
		Core::vec3 texcoord = get_texcoord(hit_item.shape, rray.phit, hit_item.pmodel->s_type);
		return Texture::get_color(texcoord, tex);
	}

	RUN_ON_GPU
	bool detect_hit(const world& models, ray& ray, hit& hit_item)
	{
		double tnear = INFINITY;
		bool hit = false;
		for (unsigned m = 0; m < models.size; m++)
		{	
			bool has_hit = false;
			if (models.models[m].s_type == Core::shape_type::TRIANGLE)
			{
				has_hit = Triangle::detect_hit(models.models[m], ray, hit_item, tnear);
			}
			else if (models.models[m].s_type == Core::shape_type::SPHERE)
			{
				has_hit = Sphere::detect_hit(models.models[m], ray, hit_item, tnear);
			}
			else if (models.models[m].s_type == Core::shape_type::BOX)
			{
				has_hit = Box::detect_hit(models.models[m], ray, hit_item, tnear);
			}
			if (has_hit) hit = has_hit;
		}
		if (!hit) return hit;
		ray.phit = ray.origin + (ray.dir * tnear);
		if (hit_item.pmodel->s_type == Core::shape_type::TRIANGLE)
		{
			triangle* ptriangle = (triangle*)hit_item.shape;
			ray.nhit = ptriangle->normal;
		}
		else if (hit_item.pmodel->s_type == Core::shape_type::SPHERE)
		{
			sphere* psphere = (sphere*)hit_item.shape;
			ray.nhit = ray.phit - psphere->center;
			if (dot(ray.nhit, ray.dir) > 0.0) ray.nhit = -ray.nhit;
		}
		else if (hit_item.pmodel->s_type == Core::shape_type::BOX)
		{
			Core::box* pbox = (Core::box*)hit_item.shape;
			ray.nhit = Box::calculate_normal(pbox, ray.phit);
			if (dot(ray.nhit, ray.dir) > 0.0) ray.nhit = -ray.nhit;
		}
		normalize(ray.nhit);
		return hit;
	}

	RUN_ON_GPU
	double get_glow_by_shape(const model& model, const ray& shadow_ray, double& t0)
	{
		double glow = 1.0;
		if (model.s_type == Core::shape_type::TRIANGLE)
		{
			triangle* triangles = (triangle*)model.dshapes;
			for (unsigned i = 0; i < model.shapes_size; i++)
			{
				if (Triangle::does_intersect(triangles[i], shadow_ray, t0))
				{
					glow = model.transparency;
					break;
				}
			}
		}
		else if (model.s_type == Core::shape_type::SPHERE)
		{
			sphere* spheres = (sphere*)model.dshapes;
			for (unsigned i = 0; i < model.shapes_size; i++)
			{
				if (Sphere::does_intersect(spheres[i], shadow_ray, t0))
				{
					glow = model.transparency;
					break;
				}
			}
		}
		else if (model.s_type == Core::shape_type::BOX)
		{
			Core::box* boxes = (Core::box*)model.dshapes;
			for (unsigned i = 0; i < model.shapes_size; i++)
			{
				if (Box::does_intersect(boxes[i], shadow_ray, t0))
				{
					glow = model.transparency;
					break;
				}
			}
		}
		return glow;
	}

	RUN_ON_GPU
	Core::vec3 get_background_reflection(const hit& hit_item, const ray& rray, const Core::cubemap* pcubemap)
	{
		Core::vec3 dir;
		if (hit_item.pmodel->s_type == Core::shape_type::TRIANGLE)
		{
			triangle* ptriangle = (triangle*)hit_item.shape;
			Core::vec3 centroid;
			centroid.x = (ptriangle->a.x + ptriangle->b.x + ptriangle->c.x) / 3.0;
			centroid.y = (ptriangle->a.y + ptriangle->b.y + ptriangle->c.y) / 3.0;
			centroid.z = (ptriangle->a.z + ptriangle->b.z + ptriangle->c.z) / 3.0;
			Core::vec3 displaced_centroid = centroid - rray.nhit;
			dir = rray.phit - displaced_centroid;
		}
		else if (hit_item.pmodel->s_type == Core::shape_type::SPHERE)
		{
			dir = rray.nhit;
		}
		else if (hit_item.pmodel->s_type == Core::shape_type::BOX)
		{
			Core::box* pbox = (Core::box*)hit_item.shape;
			dir = rray.phit - pbox->center;
		}
		normalize(dir);
		return get_background_color(pcubemap, dir);
	}
}