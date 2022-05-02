#pragma once
#include "EngineCore.cuh"
#include "Maths.cuh"

namespace Engine
{
	RUN_ON_CPU_AND_GPU
	static double dot(const Base::vec3& v1, const Base::vec3& v2)
	{
		return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
	}

	RUN_ON_CPU_AND_GPU
	static double length(const Base::vec3& v1)
	{
		return square_root(dot(v1, v1));
	}

	RUN_ON_CPU_AND_GPU
	static void normalize(Base::vec3& v1)
	{
		double len = length(v1);
		v1 = Base::vec3{ v1.x / len, v1.y / len, v1.z / len };
	}

	RUN_ON_CPU_AND_GPU
	static Base::vec3 cross(const Base::vec3& v1, const Base::vec3& v2)
	{
		return Base::vec3{ v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x };
	}

	RUN_ON_CPU_AND_GPU
	static Base::vec3 operator*(const Base::vec3& v1, const double& f)
	{
		return Base::vec3{ v1.x * f, v1.y * f, v1.z * f };
	}

	RUN_ON_CPU_AND_GPU
	static Base::vec3 operator+(const Base::vec3& v1, const Base::vec3& v2)
	{
		return Base::vec3{ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };
	}

	RUN_ON_CPU_AND_GPU
	static Base::vec3 operator*(const Base::vec3& v1, const Base::vec3& v2)
	{
		return Base::vec3{ v1.x * v2.x, v1.y * v2.y, v1.z * v2.z };
	}

	RUN_ON_CPU_AND_GPU
	static Base::vec3 operator-(const Base::vec3& v1, const Base::vec3& v2)
	{
		return Base::vec3{ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };
	}

	RUN_ON_CPU_AND_GPU
	static Base::vec3 operator-(const Base::vec3& v1)
	{
		return Base::vec3{ -v1.x, -v1.y, -v1.z };
	}

	RUN_ON_CPU_AND_GPU
	static Base::vec3& operator+=(Base::vec3& v1, const Base::vec3& v2)
	{
		v1 = v1 + v2;
		return v1;
	}

	RUN_ON_CPU_AND_GPU
	static Base::vec3& operator*=(Base::vec3& v1, const Base::vec3& v2)
	{
		v1 = v1 * v2;
		return v1;
	}

	RUN_ON_CPU_AND_GPU
	static Base::vec3& operator-=(Base::vec3& v1, const Base::vec3& v2)
	{
		v1 = v1 - v2;
		return v1;
	}

	RUN_ON_CPU_AND_GPU
	static Base::vec3 get_reflect_dir(const Base::vec3& incident_dir, const Base::vec3& nhit)
	{
		Base::vec3 reflect_dir = incident_dir - (nhit * dot(incident_dir, nhit) * 2);
		normalize(reflect_dir);
		return reflect_dir;
	}

	RUN_ON_CPU_AND_GPU
	static Base::vec3 get_refract_dir(const Base::vec3& incident_dir, const Base::vec3& nhit, const bool& inside)
	{
		double ref_index_ratio = (inside) ? 1.1f : 1 / 1.1f;
		double cosine = dot(-incident_dir, nhit);
		Base::vec3 t1 = incident_dir * ref_index_ratio;
		Base::vec3 t2 = nhit * ((ref_index_ratio * cosine) - sqrt(1 - ((ref_index_ratio * ref_index_ratio) * (1 - (cosine * cosine)))));
		Base::vec3 refract_dir = t1 + t2;
		normalize(refract_dir);
		return refract_dir;
	}
}