#include "pch.h"
#include <device_launch_parameters.h>
#include "Triangle.cuh"
#include "Core.cuh"

#define RECURSION_DEPTH 5

namespace RayTracer
{
	enum class ColorType { REFLECTION, REFRACTION };
	struct ray { Core::vec3 origin, dir, phit, nhit; };
	struct hit { triangle triangle; model model; };

	RUN_ON_GPU_CALL_FROM_CPU void render(pixels pixels, models models, double fov, Projection proj_type);
	RUN_ON_GPU Core::vec3 trace(const models& models, ray ray);
	RUN_ON_GPU bool detect_hit(const models& models, ray& ray, hit& hit_item);
	RUN_ON_GPU double schlick_approximation(double cosine, double R);
	RUN_ON_GPU double max_val(double val1, double val2);
	RUN_ON_GPU Core::vec3 calculate_color(const models& models, const ray& ray, const hit& hit);
	RUN_ON_GPU Core::vec3 calculate_reflection_dir(const Core::vec3& incident_dir, const Core::vec3& nhit);
	RUN_ON_GPU Core::vec3 calculate_refraction_dir(const Core::vec3& incident_dir, const Core::vec3& nhit, const bool& inside);
	RUN_ON_GPU Core::vec3 get_color(const ColorType type, const models& models, const ray& ray);
}

void RayTracer::draw_frame(RayTracer::pixels pixels, models models, double fov, Projection proj_type)
{
	dim3 block_size(32, 32, 1);
	dim3 grid_size(pixels.width / 32, pixels.height / 32, 1);
	render << < grid_size, block_size >> > (pixels, models, fov, proj_type);
}

RUN_ON_GPU_CALL_FROM_CPU
void RayTracer::render(RayTracer::pixels pixels, const models models, double fov, Projection proj_type)
{
	double aspect_ratio = pixels.width / pixels.height;
	double tan_val = tan((fov * 3.141592653589793) / 360.0);
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	double near_plane = (proj_type == Projection::PERSPECTIVE) ? 1.0 : 8.0;
	int index = (ty * pixels.width) + tx;
	double x = ((2.0 * ((tx + 0.5) / pixels.width)) - 1) * aspect_ratio * tan_val * near_plane;
	double y = (1.0 - (2.0 * ((ty + 0.5f) / pixels.height))) * tan_val * near_plane;
	Core::vec3 dir = (proj_type == Projection::PERSPECTIVE) ? Core::vec3{ x, y, -near_plane } : Core::vec3{ 0.0, 0.0, -near_plane };
	normalize(dir);
	Core::vec3 origin = (proj_type == Projection::PERSPECTIVE) ? Core::vec3{} : Core::vec3{ x, y };
	pixels.data[index] = trace(models, ray{ origin, dir });
}

RUN_ON_GPU
Core::vec3 RayTracer::trace(const models& models, ray ray)
{
	Core::vec3 surface_color, background{ 1.0, 1.0, 1.0 };
	hit hit_item;
	if (!detect_hit(models, ray, hit_item)) return background;
	if (hit_item.model.reflectivity > 0.0 || hit_item.model.transparency > 0.0)
	{
		Core::vec3 reflection_color = (hit_item.model.reflectivity > 0.0) ? get_color(ColorType::REFLECTION, models, ray) : background;
		Core::vec3 refraction_color = (hit_item.model.transparency > 0.0) ? get_color(ColorType::REFRACTION, models, ray) : background;
		double fresnel = schlick_approximation(dot(-ray.dir, ray.nhit), 0.1);
		surface_color = (reflection_color * fresnel + refraction_color * (1 - fresnel) * hit_item.model.transparency) * get_rgb(hit_item.triangle, ray.phit, hit_item.model.texture_data);
	}
	else
	{
		surface_color = calculate_color(models, ray, hit_item);
	}
	return surface_color;
}

RUN_ON_GPU
bool RayTracer::detect_hit(const models& models, ray& ray, hit& hit_item)
{
	double t0 = INFINITY, tnear = INFINITY;
	bool hit = false;
	for (unsigned m = 0; m < models.size; m++)
	{
		triangle* triangles = models.models[m].dtriangles;
		unsigned triangles_count = models.models[m].triangles_size;
		for (unsigned i = 0; i < triangles_count; i++)
		{
			if (does_intersect(triangles[i], ray.origin, ray.dir, t0))
			{
				if (tnear > t0)
				{
					tnear = t0;
					hit_item.triangle = triangles[i];
					hit_item.model = models.models[m];
					if (!hit) hit = true;
				}
			}
		}
	}
	if (!hit) return hit;
	ray.phit = ray.origin + (ray.dir * tnear);
	ray.nhit = hit_item.triangle.normal;
	normalize(ray.nhit);
	return hit;
}

RUN_ON_GPU
double RayTracer::schlick_approximation(double cosine, double R)
{
	return R + ((1 - R) * pow(1 - cosine, 3));
}

RUN_ON_GPU
double RayTracer::max_val(double val1, double val2)
{
	return (val1 > val2) ? val1 : val2;
}

RUN_ON_GPU
Core::vec3 RayTracer::calculate_color(const models& models, const ray& ray, const hit& hit)
{
	Core::vec3 light_pos;
	double bias = 1e-4, t0 = INFINITY;
	double glow = 1.0f;
	Core::vec3 surface_color;
	Core::vec3 shadow_dir = light_pos - ray.phit;
	normalize(shadow_dir);
	Core::vec3 shadow_origin = ray.phit + ray.nhit * bias;
	for (unsigned m = 0; m < models.size; m++)
	{
		bool is_occluded = false;
		triangle* triangles = models.models[m].dtriangles;
		unsigned triangles_count = models.models[m].triangles_size;
		for (unsigned i = 0; i < triangles_count; i++)
		{
			if (does_intersect(triangles[i], shadow_origin, shadow_dir, t0))
			{
				glow = 0.0f;
				is_occluded = true;
				break;
			}
		}
		if (is_occluded) break;
	}
	//return (get_rgb(hit.triangle, ray.phit, hit.model.texture_data) * glow * max_val(0.0, dot(ray.nhit, shadow_dir))) + hit.triangle.emissive_color;
	return (Core::vec3{ 0.5294, 0.8078, 0.9216 } * glow * max_val(0.0, dot(ray.nhit, shadow_dir))) + hit.triangle.emissive_color;
}

RUN_ON_GPU
Core::vec3 RayTracer::calculate_reflection_dir(const Core::vec3& incident_dir, const Core::vec3& nhit)
{
	Core::vec3 reflect_dir = incident_dir - (nhit * dot(incident_dir, nhit) * 2);
	normalize(reflect_dir);
	return reflect_dir;
}

RUN_ON_GPU
Core::vec3 RayTracer::calculate_refraction_dir(const Core::vec3& incident_dir, const Core::vec3& nhit, const bool& inside)
{
	double ref_index_ratio = (inside) ? 1.1f : 1 / 1.1f;
	double cosine = dot(-incident_dir, nhit);
	Core::vec3 t1 = incident_dir * ref_index_ratio;
	Core::vec3 t2 = nhit * ((ref_index_ratio * cosine) - sqrt(1 - ((ref_index_ratio * ref_index_ratio) * (1 - (cosine * cosine)))));
	Core::vec3 refract_dir = t1 + t2;
	normalize(refract_dir);
	return refract_dir;
}

RUN_ON_GPU
Core::vec3 RayTracer::get_color(const ColorType type, const models& models, const ray& pray)
{
	Core::vec3 color{ 1.0, 1.0, 1.0 };
	double bias = 1e-4;
	bool inside = false;
	int depth = 0;
	hit hit_item;
	Core::vec3 new_phit, new_nhit;
	Core::vec3 new_origin = (type == ColorType::REFRACTION) ? pray.phit - pray.nhit * bias : pray.phit;
	Core::vec3 new_dir = (type == ColorType::REFRACTION) ? calculate_refraction_dir(pray.dir, pray.nhit, inside) : calculate_reflection_dir(pray.dir, pray.nhit);
	ray nray{ new_origin, new_dir, new_phit, new_nhit };
	while ((depth < RECURSION_DEPTH) && detect_hit(models, nray, hit_item))
	{
		if ((type == ColorType::REFRACTION) ? (hit_item.model.transparency > 0.0) : (hit_item.model.reflectivity > 0.0))
		{
			color = color * get_rgb(hit_item.triangle, nray.phit, hit_item.model.texture_data);
			new_dir = (type == ColorType::REFRACTION) ? calculate_refraction_dir(new_dir, new_nhit, inside) : calculate_reflection_dir(new_dir, new_nhit);
			new_origin = (type == ColorType::REFRACTION) ? new_phit - new_nhit * bias : new_phit;
			nray.dir = new_dir;
			nray.origin = new_origin;
			depth++;
		}
		else
		{
			color = color * calculate_color(models, pray, hit_item);
			break;
		}
	}
	return color;
}