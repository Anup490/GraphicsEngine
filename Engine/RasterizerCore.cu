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
	RUN_ON_GPU_CALL_FROM_CPU void render_frame(pixels pixels, const raster_input input, model* dmodel, model* dlights, unsigned lights_count);
	RUN_ON_GPU bool is_visible(const Base::vec3& p);
	RUN_ON_GPU Base::vec3 to_raster(const pixels& pixels, const Base::vec3& ndc);
	RUN_ON_GPU Base::vec3 calculate_color(triangle* ptriangle, model* pmodel, model* dlights, unsigned lights_count);
}

void Engine::draw_background(pixels pixels, Base::mat4 dirmatrix, Base::cubemap* dcubemap)
{
	dim3 block_size(32, 32, 1);
	dim3 grid_size(pixels.width / 32, pixels.height / 32, 1);
	render_background << < grid_size, block_size >> > (pixels, dirmatrix, dcubemap);
}

void Engine::draw_frame(pixels pixels, const raster_input& input, model_data data, model* dlights, unsigned lights_count)
{
	unsigned threads = nearest_high_multiple(data.shape_count, 32);
	unsigned threads_per_block = (threads < 1024) ? threads : 1024;
	unsigned blocks = threads/1024 + 1;
	dim3 block_size(threads_per_block, 1, 1);
	dim3 grid_size(blocks, 1, 1);
	render_frame << < grid_size, block_size >> > (pixels, input, data.dmodel, dlights, lights_count);
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
}

RUN_ON_GPU_CALL_FROM_CPU 
void Engine::render_frame(pixels pixels, const raster_input input, model* dmodel, model* dlights, unsigned lights_count)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= (dmodel->shapes_size)) return;
	triangle* ptriangles = (triangle*)dmodel->dshapes;
	triangle* ptriangle = ptriangles + index;
	Base::vec3 a_ndc = input.projection * (input.view * ptriangle->a);
	Base::vec3 b_ndc = input.projection * (input.view * ptriangle->b);
	Base::vec3 c_ndc = input.projection * (input.view * ptriangle->c);
	if (!is_visible(a_ndc) && !is_visible(b_ndc) && !is_visible(c_ndc)) return;
	Base::vec3 a_raster = to_raster(pixels, a_ndc);
	Base::vec3 b_raster = to_raster(pixels, b_ndc);
	Base::vec3 c_raster = to_raster(pixels, c_ndc);
	int min_raster_x = minimum(a_raster.x, b_raster.x, c_raster.x);
	int min_raster_y = minimum(a_raster.y, b_raster.y, c_raster.y);
	int max_raster_x = maximum(a_raster.x, b_raster.x, c_raster.x);
	int max_raster_y = maximum(a_raster.y, b_raster.y, c_raster.y);
	triangle t_raster{ a_raster, b_raster, c_raster };
	for (int j = min_raster_y; j <= max_raster_y; j++)
	{
		if (j >= 0 && j < pixels.height)
		{
			for (int i = min_raster_x; i <= max_raster_x; i++)
			{
				if (i >= 0 && i < pixels.width)
				{
					if (Triangle::is_inside(t_raster, Base::vec3{ double(i), double(j) }))
					{
						Base::vec3 color = calculate_color(ptriangle, dmodel, dlights, lights_count);
						pixels.data[j * pixels.width + i] = rgb{ unsigned char(color.x * 255), unsigned char(color.y * 255), unsigned char(color.z * 255) };
					}
				}
			}
		}
	}
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
Base::vec3 Engine::calculate_color(triangle* ptriangle, model* pmodel, model* dlights, unsigned lights_count)
{
	Base::vec3 color = Texture::get_color(ptriangle->a_tex, pmodel->diffuse);
	Base::vec3 specularity; 
	if (!pmodel->specular.ptextures) specularity = Base::vec3{ pmodel->smoothness, pmodel->smoothness, pmodel->smoothness };
	else specularity = Texture::get_color(ptriangle->a_tex, pmodel->specular);
	Base::vec3 light_vec;
	for (unsigned i = 0; i < lights_count; i++)
	{
		light_vec = dlights->position - ptriangle->a;
		double light_triangle_dot = dot(light_vec, ptriangle->normal);
		color += (color * max_val(0.0, light_triangle_dot));
	}
	return get_clamped(color);
}