#pragma once
#include "PrivateBase.cuh"
#include <math.h>

namespace RayTracer
{
	RUN_ON_CPU_AND_GPU
	double square_root(double num)
	{
		return sqrt(num);
	}

	RUN_ON_CPU_AND_GPU
	double get_infinity()
	{
		return INFINITY;
	}

	RUN_ON_CPU_AND_GPU
	double max_val(double val1, double val2)
	{
		return (val1 > val2) ? val1 : val2;
	}
	
	RUN_ON_CPU_AND_GPU
	double schlick_approximation(double cosine, double R)
	{
		return R + ((1 - R) * pow(1 - cosine, 3));
	}

	RUN_ON_CPU_AND_GPU
	Core::vec3 get_clamped(const Core::vec3& unclamped)
	{
		Core::vec3 clamped;
		clamped.x = (unclamped.x > 1.0) ? 1.0 : unclamped.x;
		clamped.y = (unclamped.y > 1.0) ? 1.0 : unclamped.y;
		clamped.z = (unclamped.z > 1.0) ? 1.0 : unclamped.z;
		return clamped;
	}
}

