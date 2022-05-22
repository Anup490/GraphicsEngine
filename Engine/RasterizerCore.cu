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
	RUN_ON_GPU_CALL_FROM_CPU void transform_triangles(pixels pixels, const raster_input input, model_data* ddata);
	RUN_ON_GPU_CALL_FROM_CPU void render_frame(pixels pixels, const raster_input input, model_data* ddata);
	RUN_ON_GPU bool cull_back_face(const triangle* ptriangle);
	RUN_ON_GPU bool is_visible(const Base::vec3& p);
	RUN_ON_GPU Base::vec3 to_raster(const pixels& pixels, const Base::vec3& ndc);
	RUN_ON_GPU Base::vec3 calculate_color(const Base::vec3& texcoord, const Base::mat4& view_mat, const Base::vec3& p, model_data* ddata);
}

void Engine::draw_background(pixels pixels, Base::mat4 dirmatrix, Base::cubemap* dcubemap)
{
	dim3 block_size(32, 32, 1);
	dim3 grid_size(pixels.width / 32, pixels.height / 32, 1);
	render_background << < grid_size, block_size >> > (pixels, dirmatrix, dcubemap);
	cudaDeviceSynchronize();
}

void Engine::draw_frame(pixels pixels, const raster_input& input, model_data* ddata, unsigned shape_count)
{
	unsigned threads = nearest_high_multiple(shape_count, 32);
	unsigned threads_per_block = (threads < 1024) ? threads : 1024;
	unsigned blocks = threads/1024 + 1;
	dim3 block_size(threads_per_block, 1, 1);
	dim3 grid_size(blocks, 1, 1);
	transform_triangles << < grid_size, block_size >> > (pixels, input, ddata);
	cudaDeviceSynchronize();
	render_frame << < grid_size, block_size >> > (pixels, input, ddata);
	cudaDeviceSynchronize();
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
void Engine::transform_triangles(pixels pixels, const raster_input input, model_data* ddata)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= (ddata->dmodel->shapes_size)) return;
	triangle* ptriangles = (triangle*)ddata->dmodel->dshapes;
	ddata->dtview[index].a = input.view * ptriangles[index].a;
	ddata->dtview[index].b = input.view * ptriangles[index].b;
	ddata->dtview[index].c = input.view * ptriangles[index].c;
	Triangle::make_triangle(ddata->dtview[index].a, ddata->dtview[index].b, ddata->dtview[index].c, ddata->dtview[index]);
	ddata->dtndc[index].a = input.projection * ddata->dtview[index].a;
	ddata->dtndc[index].b = input.projection * ddata->dtview[index].b;
	ddata->dtndc[index].c = input.projection * ddata->dtview[index].c;
	ddata->dtraster[index].a = to_raster(pixels, ddata->dtndc[index].a);
	ddata->dtraster[index].b = to_raster(pixels, ddata->dtndc[index].b);
	ddata->dtraster[index].c = to_raster(pixels, ddata->dtndc[index].c);
	Triangle::make_triangle(ddata->dtraster[index].a, ddata->dtraster[index].b, ddata->dtraster[index].c, ddata->dtraster[index]);
}

RUN_ON_GPU_CALL_FROM_CPU
void Engine::render_frame(pixels pixels, const raster_input input, model_data* ddata)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= (ddata->dmodel->shapes_size)) return;
	triangle* ptriangles = (triangle*)ddata->dmodel->dshapes;
	if (cull_back_face(ddata->dtview + index)) return;
	if (!is_visible(ddata->dtndc[index].a) && !is_visible(ddata->dtndc[index].b) && !is_visible(ddata->dtndc[index].c)) return;
	int min_raster_x = minimum(ddata->dtraster[index].a.x, ddata->dtraster[index].b.x, ddata->dtraster[index].c.x);
	int min_raster_y = minimum(ddata->dtraster[index].a.y, ddata->dtraster[index].b.y, ddata->dtraster[index].c.y);
	int max_raster_x = maximum(ddata->dtraster[index].a.x, ddata->dtraster[index].b.x, ddata->dtraster[index].c.x);
	int max_raster_y = maximum(ddata->dtraster[index].a.y, ddata->dtraster[index].b.y, ddata->dtraster[index].c.y);
	for (int j = min_raster_y; j <= max_raster_y; j++)
	{
		if (j >= 0 && j < pixels.height)
		{
			for (int i = min_raster_x; i <= max_raster_x; i++)
			{
				if (i >= 0 && i < pixels.width)
				{
					Base::vec3 raster_coord{ double(i), double(j) };
					if (Triangle::is_inside(ddata->dtraster[index], raster_coord))
					{
						Base::vec3 p = Triangle::interpolate_point(ddata->dtraster[index], ddata->dtview[index], raster_coord);
						Base::vec3 texcoord = Triangle::interpolate_texcoord(ddata->dtraster[index], ddata->dtview[index], &ptriangles[index], raster_coord, p.z);
						Base::vec3 color = calculate_color(texcoord, input.view, p, ddata);
						int index = j * pixels.width + i;
						if (p.z < pixels.depth[index])
						{
							pixels.data[index] = rgb{ unsigned char(color.x * 255), unsigned char(color.y * 255), unsigned char(color.z * 255) };
							pixels.depth[index] = p.z;
						}
					}
				}
			}
		}
	}
}

RUN_ON_GPU
bool Engine::cull_back_face(const triangle* ptriangle)
{
	Base::vec3 view_dir = ptriangle->a;
	normalize(view_dir);
	return dot(view_dir, ptriangle->normal) >= 0.0;
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
Base::vec3 Engine::calculate_color(const Base::vec3& texcoord, const Base::mat4& view_mat, const Base::vec3& p, model_data* ddata)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	Base::vec3 color;
	Base::vec3 diffuse_color = Texture::get_color(texcoord, ddata->dmodel->diffuse);
	Base::vec3 specularity;
	Base::vec3 ambient_color{ 0.25, 0.25, 0.25 };
	Base::vec3 ambient = diffuse_color * ambient_color;
	if (!ddata->dmodel->specular.ptextures) specularity = Base::vec3{ ddata->dmodel->smoothness, ddata->dmodel->smoothness, ddata->dmodel->smoothness };
	else specularity = Texture::get_color(texcoord, ddata->dmodel->specular);
	for (unsigned i = 0; i < ddata->lights_count; i++)
	{
		Base::vec3 view_light_pos = view_mat * ddata->dlights->position;
		Base::vec3 light_dir = view_light_pos - p;
		normalize(light_dir);
		double light_triangle_dot = max_val(0, dot(light_dir, ddata->dtview[index].normal));
		Base::vec3 diffuse = diffuse_color * light_triangle_dot;
		Base::vec3 reflect_dir = get_reflect_dir(-light_dir, ddata->dtview[index].normal);
		normalize(reflect_dir);
		Base::vec3 view_dir = -p;
		normalize(view_dir);
		Base::vec3 specular = specularity * light_triangle_dot * pow(max_val(0.0, dot(view_dir, reflect_dir)), to_1_to_256(specularity.x));
		color += diffuse + specular;
	}
	return get_clamped(color + ambient);
}