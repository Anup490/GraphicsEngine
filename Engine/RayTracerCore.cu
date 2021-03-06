#include "pch.h"
#include <device_launch_parameters.h>
#include "RayTracerCore.cuh"
#include "RayTracerInterface.cuh"
#include "Matrix.cuh"

#define RECURSION_DEPTH 5

namespace Engine
{
	RUN_ON_GPU model* pcamera = 0;

	enum class ColorType { REFLECTION, REFRACTION };

	RUN_ON_GPU_CALL_FROM_CPU void render(pixels pixels, const world* dworld, const raytrace_input input);
	RUN_ON_GPU Base::vec3 cast_primary_ray(pixels& pixels, int index, const world& models, ray& ray);
	RUN_ON_GPU Base::vec3 cast_second_ray(const ColorType type, const world& models, const hit& first_hit, ray& ray);
	RUN_ON_GPU Base::vec3 cast_shadow_ray(const world& models, const hit& hit, ray& rray);
	RUN_ON_GPU model* get_camera(const world* dworld);
	RUN_ON_GPU double get_glow(const unsigned light_index, const world& models, const ray& shadow_ray);
}

void Engine::draw_frame(Engine::pixels pixels, const world* dworld, const raytrace_input& input)
{
	dim3 block_size(32, 32, 1);
	dim3 grid_size(pixels.width / 32, pixels.height / 32, 1);
	render << < grid_size, block_size >> > (pixels, dworld, input);
}

RUN_ON_GPU_CALL_FROM_CPU
void Engine::render(Engine::pixels pixels, const world* dworld, const raytrace_input input)
{
	double aspect_ratio = pixels.width / pixels.height;
	double tan_val = tangent(input.fov / 2.0);
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int index = (ty * pixels.width) + tx;
	double x = ((2.0 * ((tx + 0.5) / pixels.width)) - 1.0) * aspect_ratio * tan_val * input.near;
	double y = (1.0 - (2.0 * ((ty + 0.5) / pixels.height))) * tan_val * input.near;
	if(!pcamera) pcamera = get_camera(dworld);
	Base::vec3 dir = (input.proj_type == Projection::PERSPECTIVE) ? Base::vec3{ x, y, -input.near } : Base::vec3{ 0.0, 0.0, -input.near };
	normalize(dir);
	Base::vec3 origin = (input.proj_type == Projection::PERSPECTIVE) ? Base::vec3{} : Base::vec3{ x, y };
	if (input.proj_type == Projection::PERSPECTIVE) dir = input.rotator * dir;
	origin = pcamera->position;
	ray pray{ origin, dir };
	pixels.depth[index] = get_infinity();
	Base::vec3 color = cast_primary_ray(pixels, index, *dworld, pray);
	pixels.data[index] = rgb{ unsigned char(color.x * 255.0), unsigned char(color.y * 255.0), unsigned char(color.z * 255.0) };
}

RUN_ON_GPU
Base::vec3 Engine::cast_primary_ray(pixels& pixels, int index, const world& models, ray& ray)
{
	Base::vec3 surface_color{};
	hit hit_item;
	if (!detect_hit(models, ray, hit_item)) return get_background_color(models.dcubemap, ray.dir);
	pixels.depth[index] = length(pcamera->position - ray.phit);
	if (hit_item.pmodel->smoothness > 0.0 || hit_item.pmodel->transparency > 0.0)
	{
		Base::vec3 reflect_color = (hit_item.pmodel->smoothness > 0.0) ? cast_second_ray(ColorType::REFLECTION,models,hit_item,ray) : Base::vec3{};
		Base::vec3 refract_color = (hit_item.pmodel->transparency > 0.0) ? cast_second_ray(ColorType::REFRACTION,models,hit_item,ray) : Base::vec3{};
		Base::vec3 diffuse_color = get_color(hit_item, hit_item.pmodel->diffuse, ray);
		Base::vec3 dir = pcamera->position - ray.phit;
		normalize(dir);
		double fresnel = schlick_approximation(max_val(0.0, dot(dir, ray.nhit)), 0.1);
		surface_color = ((reflect_color * fresnel) + (refract_color * (1 - fresnel) * hit_item.pmodel->transparency)) * diffuse_color;
	}
	surface_color += cast_shadow_ray(models, hit_item, ray);
	return get_clamped(surface_color);
}

RUN_ON_GPU
Base::vec3 Engine::cast_second_ray(const ColorType type, const world& models, const hit& first_hit, ray& pray)
{
	Base::vec3 color{ 1.0, 1.0, 1.0 };
	double bias = 1e-4;
	bool inside = false, not_calc_color = true;
	int depth = 0;
	hit hit_item;
	ray nray;
	nray.origin = (type == ColorType::REFRACTION) ? pray.phit - pray.nhit * bias : pray.phit + pray.nhit * bias;
	nray.dir = (type == ColorType::REFRACTION) ? get_refract_dir(pray.dir, pray.nhit, inside) : get_reflect_dir(pray.dir, pray.nhit);
	while ((depth < RECURSION_DEPTH) && detect_hit(models, nray, hit_item))
	{
		if (not_calc_color) not_calc_color = false;
		if ((type == ColorType::REFRACTION) ? (hit_item.pmodel->transparency > 0.0) : (hit_item.pmodel->smoothness > 0.0))
		{
			color *= get_color(hit_item, hit_item.pmodel->diffuse, nray);
			nray.dir = (type == ColorType::REFRACTION) ? get_refract_dir(nray.dir, nray.nhit, inside) : get_reflect_dir(nray.dir, nray.nhit);
			nray.origin = (type == ColorType::REFRACTION) ? nray.phit - nray.nhit * bias : nray.phit + nray.nhit * bias;
			depth++;
		}
		else
		{
			color *= cast_shadow_ray(models, hit_item, pray);
			break;
		}
	}
	if ((type == ColorType::REFLECTION) && not_calc_color)
		color = get_background_reflection(first_hit, pray, models.dcubemap) * first_hit.pmodel->smoothness;
	return color;
}

RUN_ON_GPU
Base::vec3 Engine::cast_shadow_ray(const world& models, const hit& hit, ray& rray)
{
	Base::vec3 color;
	Base::vec3 diffuse_color = get_color(hit, hit.pmodel->diffuse, rray);
	Base::vec3 specular_color = get_specular_val(hit, hit.pmodel->specular, rray);
	Base::vec3 ambient_color = Base::vec3{ 0.25, 0.25, 0.25 };
	Base::vec3 ambient = diffuse_color * ambient_color;
	double bias = 1e-4;
	for (unsigned l = 0; l < models.size; l++)
	{
		model* light_model = &models.models[l];
		if ((light_model->m_type == Base::model_type::LIGHT) && (hit.pmodel->m_type == Base::model_type::OBJECT))
		{
			Base::vec3 shadow_dir = light_model->position - rray.phit;
			normalize(shadow_dir);
			Base::vec3 shadow_origin = rray.phit + rray.nhit * bias;
			ray shadow_ray{ shadow_origin, shadow_dir };
			double shadow_normal_dot = max_val(0.0, dot(rray.nhit, shadow_dir));
			Base::vec3 diffuse = diffuse_color * shadow_normal_dot * (1 - hit.pmodel->metallicity);
			Base::vec3 reflect_dir = get_reflect_dir(-shadow_dir, rray.nhit);
			normalize(reflect_dir);
			Base::vec3 view_dir = pcamera->position - rray.phit;
			normalize(view_dir);																
			Base::vec3 specular = specular_color * pow(max_val(0.0, dot(view_dir, reflect_dir)), to_1_to_256(specular_color.x)) * shadow_normal_dot;
			color += (diffuse + specular) * get_glow(l, models, shadow_ray) * light_model->emissive_color;
		}
	}
	return color + hit.pmodel->emissive_color + ambient;
}

RUN_ON_GPU
Engine::model* Engine::get_camera(const world* dworld)
{
	model* pcamera = 0;
	for (unsigned i = 0; i < dworld->size; i++)
	{
		model* pmodel = &(dworld->models)[i];
		if (pmodel->m_type == Base::model_type::CAMERA)
		{
			pcamera = pmodel;
			break;
		}
	}
	return pcamera;
}

RUN_ON_GPU
double Engine::get_glow(const unsigned light_index, const world& models, const ray& shadow_ray)
{
	double t0 = INFINITY, glow = 1.0;
	for (unsigned m = 0; m < models.size; m++)
	{
		if (m != light_index)
		{
			glow *= get_glow_by_shape(models.models[m], shadow_ray, t0);
			if (glow == 0.0) break;
		}
	}
	return glow;
}