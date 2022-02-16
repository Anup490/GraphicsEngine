#pragma once
#include "Core.h"

namespace RayTracer
{
	double dot(const Core::vec3& v1, const Core::vec3& v2);
	double length(const Core::vec3& v1);
	void normalize(Core::vec3& v1);
	Core::vec3 cross(const Core::vec3& v1, const Core::vec3& v2);
	Core::vec3 operator*(const Core::vec3& v1, const double& f);
	Core::vec3 operator+(const Core::vec3& v1, const Core::vec3& v2);
	Core::vec3 operator*(const Core::vec3& v1, const Core::vec3& v2);
	Core::vec3 operator-(const Core::vec3& v1, const Core::vec3& v2);
	Core::vec3 operator-(const Core::vec3& v1);
	Core::vec3& operator+=(Core::vec3& v1, const Core::vec3& v2);
	Core::vec3& operator*=(Core::vec3& v1, const Core::vec3& v2);
	Core::vec3& operator-=(Core::vec3& v1, const Core::vec3& v2);
}

