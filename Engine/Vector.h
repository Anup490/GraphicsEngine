#pragma once
#include "Base.h"

namespace Engine
{
	double dot(const Base::vec3& v1, const Base::vec3& v2);
	double length(const Base::vec3& v1);
	void normalize(Base::vec3& v1);
	Base::vec3 cross(const Base::vec3& v1, const Base::vec3& v2);
	Base::vec3 operator*(const Base::vec3& v1, const double& f);
	Base::vec3 operator+(const Base::vec3& v1, const Base::vec3& v2);
	Base::vec3 operator*(const Base::vec3& v1, const Base::vec3& v2);
	Base::vec3 operator-(const Base::vec3& v1, const Base::vec3& v2);
	Base::vec3 operator-(const Base::vec3& v1);
	Base::vec3& operator+=(Base::vec3& v1, const Base::vec3& v2);
	Base::vec3& operator*=(Base::vec3& v1, const Base::vec3& v2);
	Base::vec3& operator-=(Base::vec3& v1, const Base::vec3& v2);
}

