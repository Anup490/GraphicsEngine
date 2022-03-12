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
}

