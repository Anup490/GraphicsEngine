#pragma once
#include "PublicBase.h"
#include <cuda_runtime.h>

#define RUN_ON_CPU __host__
#define RUN_ON_GPU __device__
#define RUN_ON_CPU_AND_GPU __host__ __device__
#define RUN_ON_GPU_CALL_FROM_CPU __global__

namespace RayTracer
{
	struct texture
	{
		float* dtextures = 0;
		int width = 0;
		int height = 0;
		int channels = 0;
	};

	struct triangle
	{
		Core::vec3 a, b, c, ab, bc, ca, a_tex, b_tex, c_tex, normal;
		double plane_distance = 0.0, area = 0.0;
	};

	struct model
	{
		Core::vec3 position, emissive_color;
		triangle* dtriangles = 0;
		texture texture_data;
		unsigned triangles_size = 0;
		double reflectivity = 0.0, transparency = 0.0;
	};

	struct models
	{
		model* models;
		unsigned size = 0;
	};
}