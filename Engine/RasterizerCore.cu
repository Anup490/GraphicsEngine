#include "RasterizerCore.cuh"
#include "Maths.cuh"
#include "Vector.cuh"
#include "Matrix.cuh"
#include <device_launch_parameters.h>

namespace Engine
{
	RUN_ON_GPU_CALL_FROM_CPU void render_background(pixels pixels, raster_input* draster_input, Base::cubemap* dcubemap);
	RUN_ON_GPU_CALL_FROM_CPU void render_frame(pixels pixels, raster_input* draster_input, Engine::model* dmodel);
	RUN_ON_GPU Base::vec3 to_ndc(raster_input* draster_input, Base::vec3& p);
}

void Engine::draw_background(pixels pixels, raster_input* draster_input, Base::cubemap* dcubemap)
{
	dim3 block_size(32, 32, 1);
	dim3 grid_size(pixels.width / 32, pixels.height / 32, 1);
	render_background << < grid_size, block_size >> > (pixels, draster_input, dcubemap);
}

void Engine::draw_frame(pixels pixels, raster_input* draster_input, Engine::model* dmodel)
{
	dim3 block_size(32, 32, 1);
	dim3 grid_size(pixels.width / 32, pixels.height / 32, 1);
	render_frame << < grid_size, block_size >> > (pixels, draster_input, dmodel);
}

RUN_ON_GPU_CALL_FROM_CPU 
void Engine::render_background(pixels pixels, raster_input* draster_input, Base::cubemap* dcubemap)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	pixels.data[ty * pixels.width + tx] = rgb{255, 255, 255};
}

RUN_ON_GPU_CALL_FROM_CPU 
void Engine::render_frame(pixels pixels, raster_input* draster_input, Engine::model* dmodel)
{
}

RUN_ON_GPU 
Base::vec3 Engine::to_ndc(raster_input* draster_input, Base::vec3& p)
{
	Base::vec3 p_view_space = draster_input->view * p;
	return draster_input->projection * p_view_space;
}