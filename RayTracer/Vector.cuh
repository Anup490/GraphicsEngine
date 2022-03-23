#pragma once
#include "PrivateBase.cuh"
#include "Maths.h"

namespace RayTracer
{
	RUN_ON_CPU_AND_GPU
	double dot(const Core::vec3& v1, const Core::vec3& v2)
	{
		return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
	}

	RUN_ON_CPU_AND_GPU
	double length(const Core::vec3& v1)
	{
		return square_root(dot(v1, v1));
	}

	RUN_ON_CPU_AND_GPU
	void normalize(Core::vec3& v1)
	{
		double len = length(v1);
		v1 = Core::vec3{ v1.x / len, v1.y / len, v1.z / len };
	}

	RUN_ON_CPU_AND_GPU
	Core::vec3 cross(const Core::vec3& v1, const Core::vec3& v2)
	{
		return Core::vec3{ v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x };
	}

	RUN_ON_CPU_AND_GPU
	Core::vec3 operator*(const Core::vec3& v1, const double& f)
	{
		return Core::vec3{ v1.x * f, v1.y * f, v1.z * f };
	}

	RUN_ON_CPU_AND_GPU
	Core::vec3 operator+(const Core::vec3& v1, const Core::vec3& v2)
	{
		return Core::vec3{ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };
	}

	RUN_ON_CPU_AND_GPU
	Core::vec3 operator*(const Core::vec3& v1, const Core::vec3& v2)
	{
		return Core::vec3{ v1.x * v2.x, v1.y * v2.y, v1.z * v2.z };
	}

	RUN_ON_CPU_AND_GPU
	Core::vec3 operator-(const Core::vec3& v1, const Core::vec3& v2)
	{
		return Core::vec3{ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };
	}

	RUN_ON_CPU_AND_GPU
	Core::vec3 operator-(const Core::vec3& v1)
	{
		return Core::vec3{ -v1.x, -v1.y, -v1.z };
	}

	RUN_ON_CPU_AND_GPU
	Core::vec3& operator+=(Core::vec3& v1, const Core::vec3& v2)
	{
		v1 = v1 + v2;
		return v1;
	}

	RUN_ON_CPU_AND_GPU
	Core::vec3& operator*=(Core::vec3& v1, const Core::vec3& v2)
	{
		v1 = v1 * v2;
		return v1;
	}

	RUN_ON_CPU_AND_GPU
	Core::vec3& operator-=(Core::vec3& v1, const Core::vec3& v2)
	{
		v1 = v1 - v2;
		return v1;
	}
}