#include "math.h"

namespace RayTracer
{
	struct vec3
	{
		double x, y, z;
		double length;
		vec3() :x(0.0f), y(0.0f), z(0.0f), length(sqrt(0.0)) {}
		vec3(double f) : x(f), y(f), z(f), length(sqrt(3.0 * f * f)) {}
		vec3(double x, double y, double z) : x(x), y(y), z(z), length(sqrt(x * x + y * y + z * z)) {}
	};

	double dot(const vec3& v1, const vec3& v2)
	{
		return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
	}

	vec3 cross(const vec3& v1, const vec3& v2)
	{
		return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
	}

	void normalize(vec3& v1)
	{
		v1 = vec3(v1.x / v1.length, v1.y / v1.length, v1.z / v1.length);
	}

	vec3 operator*(const vec3& v1, const double& f)
	{
		return vec3(v1.x * f, v1.y * f, v1.z * f);
	}

	vec3 operator+(const vec3& v1, const vec3& v2)
	{
		return vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
	}

	vec3 operator*(const vec3& v1, const vec3& v2)
	{
		return vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
	}

	vec3 operator-(const vec3& v1, const vec3& v2)
	{
		return vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
	}

	vec3 operator-(const vec3& v1)
	{
		return vec3(-v1.x, -v1.y, -v1.z);
	}

	vec3& operator+=(vec3& v1, const vec3& v2)
	{
		v1 = v1 + v2;
		return v1;
	}

	vec3& operator*=(vec3& v1, const vec3& v2)
	{
		v1 = v1 * v2;
		return v1;
	}

	vec3& operator-=(vec3& v1, const vec3& v2)
	{
		v1 = v1 - v2;
		return v1;
	}
}