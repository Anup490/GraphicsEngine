#include "pch.h"
#include <device_launch_parameters.h>
#include "Core.cuh"
#include "Interface.cuh"
#include "Matrix.cuh"

#define RECURSION_DEPTH 5

namespace RayTracer
{
	enum class ColorType { REFLECTION, REFRACTION };

	RUN_ON_GPU_CALL_FROM_CPU void render(pixels pixels, models models, double fov, Projection proj_type);
	RUN_ON_GPU Core::vec3 cast_primary_ray(const models& models, ray& ray);
	RUN_ON_GPU Core::vec3 cast_second_ray(const ColorType type, const models& models, ray& ray);
	RUN_ON_GPU Core::vec3 get_reflect_dir(const Core::vec3& incident_dir, const Core::vec3& nhit);
	RUN_ON_GPU Core::vec3 get_refract_dir(const Core::vec3& incident_dir, const Core::vec3& nhit, const bool& inside);
	RUN_ON_GPU Core::vec3 cast_shadow_ray(const models& models, ray& ray, const hit& hit);
	RUN_ON_GPU model* get_camera(const models& models);
	RUN_ON_GPU double get_glow(const unsigned light_index, const models& models, const ray& shadow_ray);
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

	/*Core::mat4 translation;
	translation.matrix[0] = 1;
	translation.matrix[1] = 0;
	translation.matrix[2] = 0.5;
	translation.matrix[3] = 0;
	translation.matrix[4] = 1;
	translation.matrix[5] = 0.5;
	translation.matrix[6] = 0;
	translation.matrix[7] = 0;
	translation.matrix[8] = 1;
	dir = translation * dir;
	origin = translation * origin;*/

	ray pray{ origin, dir };
	Core::vec3 color = cast_primary_ray(models, pray);
	pixels.data[index] = rgb{ unsigned char(color.x * 255.0), unsigned char(color.y * 255.0), unsigned char(color.z * 255.0) };
}

RUN_ON_GPU
Core::vec3 RayTracer::cast_primary_ray(const models& models, ray& ray)
{
	Core::vec3 surface_color, background{ 1.0, 1.0, 1.0 };
	hit hit_item;
	if (!detect_hit(models, ray, hit_item)) return background;
	if (hit_item.pmodel->reflectivity > 0.0 || hit_item.pmodel->transparency > 0.0)
	{
		Core::vec3 reflection_color = (hit_item.pmodel->reflectivity > 0.0) ? cast_second_ray(ColorType::REFLECTION, models, ray) : background;
		Core::vec3 refraction_color = (hit_item.pmodel->transparency > 0.0) ? cast_second_ray(ColorType::REFRACTION, models, ray) : background;
		double fresnel = schlick_approximation(dot(-ray.dir, ray.nhit), 0.1);
		Core::vec3 diffuse_color = get_color(hit_item, ray, hit_item.pmodel->diffuse);
		surface_color = (reflection_color * fresnel + refraction_color * (1 - fresnel) * hit_item.pmodel->transparency) * diffuse_color;
	}
	else
	{
		surface_color = cast_shadow_ray(models, ray, hit_item);
	}
	return surface_color;
}

RUN_ON_GPU
Core::vec3 RayTracer::cast_second_ray(const ColorType type, const models& models, ray& pray)
{
	Core::vec3 color { 1.0, 1.0, 1.0 };
	double bias = 1e-4;
	bool inside = false;
	int depth = 0;
	hit hit_item;
	ray nray;
	nray.origin = (type == ColorType::REFRACTION) ? pray.phit - pray.nhit * bias : pray.phit;
	nray.dir = (type == ColorType::REFRACTION) ? get_refract_dir(pray.dir, pray.nhit, inside) : get_reflect_dir(pray.dir, pray.nhit);
	while ((depth < RECURSION_DEPTH) && detect_hit(models, nray, hit_item))
	{
		if ((type == ColorType::REFRACTION) ? (hit_item.pmodel->transparency > 0.0) : (hit_item.pmodel->reflectivity > 0.0))
		{
			color *= get_color(hit_item, nray, hit_item.pmodel->diffuse);
			nray.dir = (type == ColorType::REFRACTION) ? get_refract_dir(nray.dir, nray.nhit, inside) : get_reflect_dir(nray.dir, nray.nhit);
			nray.origin = (type == ColorType::REFRACTION) ? nray.phit - nray.nhit * bias : nray.phit;
			depth++;
		}
		else
		{
			color *= cast_shadow_ray(models, pray, hit_item);
			break;
		}
	}
	return color;
}

RUN_ON_GPU
Core::vec3 RayTracer::get_reflect_dir(const Core::vec3& incident_dir, const Core::vec3& nhit)
{
	Core::vec3 reflect_dir = incident_dir - (nhit * dot(incident_dir, nhit) * 2);
	normalize(reflect_dir);
	return reflect_dir;
}

RUN_ON_GPU
Core::vec3 RayTracer::get_refract_dir(const Core::vec3& incident_dir, const Core::vec3& nhit, const bool& inside)
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
Core::vec3 RayTracer::cast_shadow_ray(const models& models, ray& rray, const hit& hit)
{
	Core::vec3 color;
	double bias = 1e-4;
	model* pcamera = get_camera(models);
	for (unsigned l = 0; l < models.size; l++)
	{
		model* light_model = &models.models[l];
		if (light_model->m_type == Core::model_type::LIGHT)
		{
			Core::vec3 shadow_dir = light_model->position - rray.phit;
			normalize(shadow_dir);
			Core::vec3 shadow_origin = rray.phit + rray.nhit * bias;
			ray shadow_ray{ shadow_origin, shadow_dir };
			Core::vec3 diffuse = get_color(hit, rray, hit.pmodel->diffuse) * max_val(0.0, dot(rray.nhit, shadow_dir));
			Core::vec3 reflect_dir = get_reflect_dir(-shadow_dir, rray.nhit);
			normalize(reflect_dir);
			Core::vec3 view_dir = pcamera->position - rray.phit;
			normalize(view_dir);
			Core::vec3 specular = get_color(hit, rray, hit.pmodel->specular) * pow(max_val(0.0, dot(view_dir, reflect_dir)), 32);
			color += get_clamped(diffuse + specular) * get_glow(l, models, shadow_ray) * light_model->emissive_color;
		}
	}
	return color + hit.pmodel->emissive_color;
}

RUN_ON_GPU 
RayTracer::model* RayTracer::get_camera(const models& models)
{
	model* pcamera;
	for (unsigned i=0; i<models.size; i++)
	{
		model* pmodel = &models.models[i];
		if (pmodel->m_type == Core::model_type::CAMERA)
		{
			pcamera = pmodel;
			break;
		}
	}
	return pcamera;
}

RUN_ON_GPU 
double RayTracer::get_glow(const unsigned light_index, const models& models, const ray& shadow_ray)
{
	double t0 = INFINITY, glow = 1.0;
	for (unsigned m = 0; m < models.size; m++)
	{
		if (m != light_index)
		{
			glow = get_glow_val(models.models[m], shadow_ray, t0);
			if (glow == 0.0) break;
		}
	}
	return glow;
}