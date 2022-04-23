#pragma once
#include "Base.h"

namespace RayTracer
{
	Base::vec3 operator*(const Base::mat4& m, const Base::vec3& v);
	Base::mat4 operator*(const Base::mat4& m1, const Base::mat4& m2);
}