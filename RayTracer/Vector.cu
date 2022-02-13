#include "Vector.cuh"

void RayTracer::normalize(Core::vec3& v1)
{
	v1 = Core::vec3(v1.x / v1.length, v1.y / v1.length, v1.z / v1.length);
}

double RayTracer::dot(const Core::vec3& v1, const Core::vec3& v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

Core::vec3 RayTracer::cross(const Core::vec3& v1, const Core::vec3& v2)
{
	return Core::vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

Core::vec3 RayTracer::operator*(const Core::vec3& v1, const double& f)
{
	return Core::vec3(v1.x * f, v1.y * f, v1.z * f);
}

Core::vec3 RayTracer::operator+(const Core::vec3& v1, const Core::vec3& v2)
{
	return Core::vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

Core::vec3 RayTracer::operator*(const Core::vec3& v1, const Core::vec3& v2)
{
	return Core::vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

Core::vec3 RayTracer::operator-(const Core::vec3& v1, const Core::vec3& v2)
{
	return Core::vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

Core::vec3 RayTracer::operator-(const Core::vec3& v1)
{
	return Core::vec3(-v1.x, -v1.y, -v1.z);
}

Core::vec3& RayTracer::operator+=(Core::vec3& v1, const Core::vec3& v2)
{
	v1 = v1 + v2;
	return v1;
}

Core::vec3& RayTracer::operator*=(Core::vec3& v1, const Core::vec3& v2)
{
	v1 = v1 * v2;
	return v1;
}

Core::vec3& RayTracer::operator-=(Core::vec3& v1, const Core::vec3& v2)
{
	v1 = v1 - v2;
	return v1;
}