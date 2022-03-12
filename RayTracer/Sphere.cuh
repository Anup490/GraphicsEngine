#pragma once
#include "PrivateBase.cuh"
#include "Vector.h"

namespace RayTracer
{
	double square_root(double num);

	RUN_ON_GPU
	bool does_intersect(const sphere& s, const ray& r, double& t0, double& t1)
	{
		Core::vec3 l = s.center - r.origin;
		double tca = dot(l, r.dir);
		if (tca < 0) return false;
		double d_square = dot(l, l) - (tca * tca);
		if (d_square > s.radius_square) return false;
		double thc = square_root(s.radius_square - d_square);
		t0 = tca - thc;
		t1 = tca + thc;
		return true;
	}
}