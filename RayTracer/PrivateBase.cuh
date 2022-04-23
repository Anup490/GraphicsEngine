#pragma once
#include "PublicBase.h"
#include <cuda_runtime.h>

#define RUN_ON_CPU __host__
#define RUN_ON_GPU __device__
#define RUN_ON_CPU_AND_GPU __host__ __device__
#define RUN_ON_GPU_CALL_FROM_CPU __global__

namespace RayTracer
{
	struct triangle
	{
		Base::vec3 a, b, c, ab, bc, ca, a_tex, b_tex, c_tex, normal;
		double plane_distance = 0.0, area = 0.0;
	};

	struct sphere
	{
		double radius_square;
		Base::vec3 center;
	};

	struct model
	{
		Base::vec3 position, emissive_color, min_coord, max_coord, surface_color;
		double smoothness = 0.0, transparency = 0.0, metallicity = 0.0;
		Base::texture diffuse, specular;
		void* dshapes = 0;
		unsigned shapes_size = 0;
		Base::shape_type s_type;
		Base::model_type m_type;
	};

	struct world
	{
		model* models;
		unsigned size = 0;
		Base::cubemap* dcubemap;
	};

	struct ray 
	{ 
		Base::vec3 origin, dir, phit, nhit;
	};

	struct hit { void* shape; model* pmodel; };

	enum class face { LEFT, RIGHT, TOP, BOTTOM, FRONT, BACK, NONE };
}