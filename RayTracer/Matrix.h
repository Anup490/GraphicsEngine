#pragma once
#include "Core.h"

namespace RayTracer
{
	Core::vec3 operator*(const Core::mat4& m, const Core::vec3& v);
	Core::mat4 operator*(const Core::mat4& m1, const Core::mat4& m2);
}