#pragma once
#include "EngineCore.cuh"


namespace Engine
{
	namespace Triangle
	{
		double edge_function(const Base::vec3& a, const Base::vec3& b, const Base::vec3& c);
		bool is_inside(const triangle& t, const Base::vec3& p);
		bool does_intersect(const triangle& t, const ray& r, double& distance);
		Base::vec3 get_texcoord(const triangle& t, const Base::vec3& p);
		bool detect_hit(model& model, ray& ray, hit& hit_item, double& tnear);
	}
}


/*
	The RUN_ON_CPU or RUN_ON_CPU_AND_GPU macros should be used in the definitions of above function 
	or else the test will not work.
*/