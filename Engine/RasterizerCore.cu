#include "RasterizerCore.cuh"
#include "Maths.cuh"
#include "Vector.cuh"
#include "Matrix.cuh"
#include "Triangle.cuh"
#include "Background.cuh"
#include <device_launch_parameters.h>

namespace Engine
{
	RUN_ON_GPU_CALL_FROM_CPU void render_background(pixels pixels, Base::mat4 dirmatrix, Base::cubemap* dcubemap);
	RUN_ON_GPU_CALL_FROM_CPU void render_frame(pixels pixels, const raster_input input, model* dmodel, model* dcamera, model* dlights, unsigned lights_count);
	RUN_ON_GPU bool cull_back_face(const model* dcamera, const triangle* ptriangle);
	RUN_ON_GPU bool is_visible(const Base::vec3& p);
	RUN_ON_GPU Base::vec3 to_raster(const pixels& pixels, const Base::vec3& ndc);
	RUN_ON_GPU Base::vec3 calculate_color(const triangle& t_raster, const triangle& t_view, const triangle* ptriangle, const Base::vec3& raster_coord, const double& depth, model* pmodel, model* dcamera, model* dlights, unsigned lights_count);
}

void Engine::draw_background(pixels pixels, Base::mat4 dirmatrix, Base::cubemap* dcubemap)
{
	dim3 block_size(32, 32, 1);
	dim3 grid_size(pixels.width / 32, pixels.height / 32, 1);
	render_background << < grid_size, block_size >> > (pixels, dirmatrix, dcubemap);
}

void Engine::draw_frame(pixels pixels, const raster_input& input, model_data data, model* dcamera, model* dlights, unsigned lights_count)
{
	unsigned threads = nearest_high_multiple(data.shape_count, 32);
	unsigned threads_per_block = (threads < 1024) ? threads : 1024;
	unsigned blocks = threads/1024 + 1;
	dim3 block_size(threads_per_block, 1, 1);
	dim3 grid_size(blocks, 1, 1);
	render_frame << < grid_size, block_size >> > (pixels, input, data.dmodel, dcamera, dlights, lights_count);
}

RUN_ON_GPU_CALL_FROM_CPU 
void Engine::render_background(pixels pixels, Base::mat4 dirmatrix, Base::cubemap* dcubemap)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	double ndcx = ((2.0 * ((tx + 0.5) / pixels.width)) - 1.0);
	double ndcy = (1.0 - (2.0 * ((ty + 0.5) / pixels.height)));
	Base::vec3 dir = dirmatrix * Base::vec3{ ndcx, ndcy, 1 };
	Base::vec3 color = get_background_color(dcubemap, dir);
	pixels.data[ty * pixels.width + tx] = rgb{ unsigned char(color.x * 255.0), unsigned char(color.y * 255.0), unsigned char(color.z * 255.0) };
	pixels.depth[ty * pixels.width + tx] = get_infinity();
}

RUN_ON_GPU_CALL_FROM_CPU 
void Engine::render_frame(pixels pixels, const raster_input input, model* dmodel, model* dcamera, model* dlights, unsigned lights_count)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= (dmodel->shapes_size)) return;
	triangle* ptriangles = (triangle*)dmodel->dshapes;
	triangle* ptriangle = ptriangles + index;
	if (cull_back_face(dcamera, ptriangle)) return;
	Base::vec3 a_view = input.view * ptriangle->a;
	Base::vec3 b_view = input.view * ptriangle->b;
	Base::vec3 c_view = input.view * ptriangle->c;
	Base::vec3 a_ndc = input.projection * a_view;
	Base::vec3 b_ndc = input.projection * b_view;
	Base::vec3 c_ndc = input.projection * c_view;
	if (!is_visible(a_ndc) && !is_visible(b_ndc) && !is_visible(c_ndc)) return;
	triangle t_view;
	Triangle::make_triangle(a_view, b_view, c_view, t_view);
	Base::vec3 a_raster = to_raster(pixels, a_ndc);
	Base::vec3 b_raster = to_raster(pixels, b_ndc);
	Base::vec3 c_raster = to_raster(pixels, c_ndc);
	triangle t_raster;
	Triangle::make_triangle(a_raster, b_raster, c_raster, t_raster);
	int min_raster_x = minimum(t_raster.a.x, t_raster.b.x, t_raster.c.x);
	int min_raster_y = minimum(t_raster.a.y, t_raster.b.y, t_raster.c.y);
	int max_raster_x = maximum(t_raster.a.x, t_raster.b.x, t_raster.c.x);
	int max_raster_y = maximum(t_raster.a.y, t_raster.b.y, t_raster.c.y);
	for (int j = min_raster_y; j <= max_raster_y; j++)
	{
		if (j >= 0 && j < pixels.height)
		{
			for (int i = min_raster_x; i <= max_raster_x; i++)
			{
				if (i >= 0 && i < pixels.width)
				{
					Base::vec3 raster_coord{ double(i), double(j) };
					if (Triangle::is_inside(t_raster, raster_coord))
					{
						double depth = Triangle::interpolate_depth(t_raster, t_view, raster_coord);
						Base::vec3 color = calculate_color(t_raster, t_view, ptriangle, raster_coord, depth, dmodel, dcamera, dlights, lights_count);
						int index = j * pixels.width + i;
						if (depth < pixels.depth[index])
						{
							pixels.data[index] = rgb{ unsigned char(color.x * 255), unsigned char(color.y * 255), unsigned char(color.z * 255) };
							pixels.depth[index] = depth;
						}
					}
				}
			}
		}
	}
}

RUN_ON_GPU 
bool Engine::cull_back_face(const model* dcamera, const triangle* ptriangle)
{
	Base::vec3 view_dir = dcamera->position - ptriangle->a;
	normalize(view_dir);
	return dot(view_dir, ptriangle->normal) < 0.0;
}

RUN_ON_GPU 
bool Engine::is_visible(const Base::vec3& p)
{
	if (p.x < -1.0 || p.x > 1.0) return false;
	if (p.y < -1.0 || p.y > 1.0) return false;
	if (p.z < -1.0 || p.z > 1.0) return false;
	return true;
}

RUN_ON_GPU 
Base::vec3 Engine::to_raster(const pixels& pixels, const Base::vec3& ndc)
{
	Base::vec3 raster;
	raster.x = (pixels.width * (ndc.x + 1)) / 2;
	raster.y = (pixels.height * (1 - ndc.y)) / 2;
	return raster;
}

RUN_ON_GPU 
Base::vec3 Engine::calculate_color(const triangle& t_raster, const triangle& t_view, const triangle* ptriangle, const Base::vec3& raster_coord, const double& depth, model* pmodel, model* dcamera, model* dlights, unsigned lights_count)
{
	Base::vec3 texcoord = Triangle::interpolate_texcoord(t_raster, t_view, ptriangle, raster_coord, depth);
	Base::vec3 color;
	Base::vec3 diffuse_color = Texture::get_color(texcoord, pmodel->diffuse);
	Base::vec3 specularity;
	Base::vec3 ambient_color{ 0.25, 0.25, 0.25 };
	Base::vec3 ambient = diffuse_color * ambient_color;
	if (!pmodel->specular.ptextures) specularity = Base::vec3{ pmodel->smoothness, pmodel->smoothness, pmodel->smoothness };
	else specularity = Texture::get_color(texcoord, pmodel->specular);
	for (unsigned i = 0; i < lights_count; i++)
	{
		Base::vec3 light_dir = dlights->position - ptriangle->a;
		normalize(light_dir);
		double light_triangle_dot = max_val(0, dot(light_dir, ptriangle->normal));
		Base::vec3 diffuse = diffuse_color * light_triangle_dot;
		Base::vec3 reflect_dir = get_reflect_dir(-light_dir, ptriangle->normal);
		normalize(reflect_dir);
		Base::vec3 view_dir = dcamera->position - ptriangle->a;
		normalize(view_dir);
		Base::vec3 specular = specularity * light_triangle_dot * pow(max_val(0.0, dot(view_dir, reflect_dir)), to_1_to_256(specularity.x));
		color += diffuse + specular;
	}
	return get_clamped(color + ambient);
}