#pragma once
#include "PrivateBase.cuh"
#include "Matrix.h"

namespace RayTracer
{
	namespace Triangle
	{
		double edge_function(const Core::vec3& a, const Core::vec3& b, const Core::vec3& c);
		bool is_inside(const triangle& t, const Core::vec3& p);
		bool does_intersect(const triangle& t, const ray& r, double& distance);
		Core::vec3 get_texcoord(const triangle& t, const Core::vec3& p);
		bool detect_hit(model& model, ray& ray, hit& hit_item, double& tnear);
	}
}


/*
	The RUN_ON_CPU or RUN_ON_CPU_AND_GPU macros should be used in the definitions of above function 
	or else the test will not work.
*/