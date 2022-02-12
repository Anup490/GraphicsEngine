#pragma once
namespace Core
{
	struct vec3
	{
		double x, y, z;
		double length;
		vec3();
		vec3(double f);
		vec3(double x, double y, double z);
		double dot(const vec3& v) const;
		vec3 cross(const vec3& v) const;
		void normalize();
		vec3 operator*(const double& f) const;
		vec3 operator+(const vec3& v) const;
		vec3 operator*(const vec3& v) const;
		vec3 operator-(const vec3& v) const;
		vec3 operator-() const;
		vec3& operator+=(const vec3& v);
		vec3& operator*=(const vec3& v);
		vec3& operator-=(const vec3& v);
	};

	struct vertex
	{
		vec3 position;
		vec3 normal;
		vec3 texcoord;
	};

	struct triangle
	{
		vertex A, B, C;
	};

}