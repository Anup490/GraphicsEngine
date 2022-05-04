#include "EngineCore.cuh"
#include <device_launch_parameters.h>

namespace Engine
{
	RUN_ON_GPU_CALL_FROM_CPU void mix_frames(const pixels p1, const pixels p2, pixels p3);
}

void Engine::mix(const pixels& pixels1, const pixels& pixels2, rgb* prgb)
{
	if (pixels1.width != pixels2.width) return;
	if (pixels1.height != pixels2.height) return;
	pixels p1(pixels1.width, pixels1.height);
	pixels p2(pixels2.width, pixels2.height);
	pixels p3(pixels1.width, pixels2.height);
	cudaMalloc(&p1.data, sizeof(rgb) * pixels1.width * pixels1.height);
	cudaMemcpy(p1.data, pixels1.data, sizeof(rgb) * pixels1.width * pixels1.height, cudaMemcpyHostToDevice);
	cudaMalloc(&p2.data, sizeof(rgb) * pixels2.width * pixels2.height);
	cudaMemcpy(p2.data, pixels2.data, sizeof(rgb) * pixels2.width * pixels2.height, cudaMemcpyHostToDevice);
	cudaMalloc(&p3.data, sizeof(rgb) * pixels1.width * pixels2.height);
	p1.depth = pixels1.depth;
	p2.depth = pixels2.depth;
	dim3 block_size(32,32,1);
	dim3 grid_size(pixels1.width / 32, pixels2.height / 32, 1);
	mix_frames << < grid_size, block_size >> > (p1, p2, p3);
	cudaMemcpy(prgb, p3.data, sizeof(rgb) * p3.width * p3.height, cudaMemcpyDeviceToHost);
	cudaFree(p1.data);
	cudaFree(p2.data);
	cudaFree(p3.data);
}

RUN_ON_GPU_CALL_FROM_CPU 
void Engine::mix_frames(const pixels p1, const pixels p2, pixels p3)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int index = (ty * p3.width) + tx;
	if (p1.depth[index] < p2.depth[index])
		p3.data[index] = p1.data[index];
	else
		p3.data[index] = p2.data[index];
}