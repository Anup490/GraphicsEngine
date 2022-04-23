#pragma once
#include "EngineCore.cuh"
#include <math.h>

namespace Engine
{
	RUN_ON_CPU_AND_GPU
	double square_root(const double& num)
	{
		return sqrt(num);
	}

	RUN_ON_CPU_AND_GPU
	double get_infinity()
	{
		return INFINITY;
	}

	RUN_ON_CPU_AND_GPU
	double get_pie()
	{
		return 3.141592653589793;
	}

	RUN_ON_CPU_AND_GPU
	double max_val(const double& val1, const double& val2)
	{
		return (val1 > val2) ? val1 : val2;
	}
	
	RUN_ON_CPU_AND_GPU
	double schlick_approximation(const double& cosine, const double& R)
	{
		return R + ((1 - R) * pow(1 - cosine, 3));
	}

	RUN_ON_CPU_AND_GPU
	Base::vec3 get_clamped(const Base::vec3& unclamped)
	{
		Base::vec3 clamped = unclamped;
		if (unclamped.x > 1.0) clamped.x = 1.0;
		else if (unclamped.x < 0.0) clamped.x = 0.0;
		if (unclamped.y > 1.0) clamped.y = 1.0;
		else if (unclamped.y < 0.0) clamped.y = 0.0;
		if (unclamped.z > 1.0) clamped.z = 1.0;
		else if (unclamped.z < 0.0) clamped.z = 0.0;
		return clamped;
	}

	RUN_ON_CPU_AND_GPU
	double to_radian(const double& d)
	{
		return (d * get_pie()) / 180.0;
	}

	RUN_ON_CPU_AND_GPU
	double sine(const double& angle)
	{
		return sin(to_radian(angle));
	}

	RUN_ON_CPU_AND_GPU
	double cosine(const double& angle)
	{
		return cos(to_radian(angle));
	}

	RUN_ON_CPU_AND_GPU
	double tangent(const double& angle)
	{
		return tan(to_radian(angle));
	}

	RUN_ON_CPU_AND_GPU
	double minimum(const double& a, const double& b, const double& c)
	{
		double m = (a < b) ? a : b;
		return (m < c) ? m : c;
	}

	RUN_ON_CPU_AND_GPU
	double maximum(const double& a, const double& b, const double& c)
	{
		double m = (a > b) ? a : b;
		return (m > c) ? m : c;
	}

	RUN_ON_CPU_AND_GPU
	void swap(double& a, double& b)
	{
		double temp = a;
		a = b;
		b = temp;
	}

	RUN_ON_CPU_AND_GPU
	bool equal(const double& d1, const double& d2)
	{
		double diff = (d1 > d2) ? d1 - d2 : d2 - d1;
		return diff < 0.000001;
	}

	RUN_ON_CPU_AND_GPU
	double to_1_to_256(const double& smoothness)
	{
		return (255.0 * smoothness) + 1.0;
	}
}

