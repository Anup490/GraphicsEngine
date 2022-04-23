#pragma once
#include "EngineCore.cuh"
#include "Vector.cuh"

namespace Engine
{
	namespace Triangle
	{
		RUN_ON_CPU_AND_GPU
		static double edge_function(const Base::vec3& a, const Base::vec3& b, const Base::vec3& c)
		{
			return ((a.x - b.x) * (c.y - a.y)) - ((a.y - b.y) * (c.x - a.x));
		}

		RUN_ON_CPU_AND_GPU
		static bool is_inside(const triangle& t, const Base::vec3& p)
		{
			return (edge_function(t.c, t.a, p) < 0.0f) && (edge_function(p, t.a, t.b) < 0.0f) && (edge_function(t.c, p, t.b) < 0.0f);
		}

		RUN_ON_CPU_AND_GPU
		static bool does_intersect(const triangle& t, const ray& r, double& distance)
		{
			double dir_normal_dot = dot(r.dir, t.normal);
			if (dir_normal_dot >= 0.0f) return false;
			double origin_normal_dot = dot(r.origin, t.normal);
			distance = ((t.plane_distance + origin_normal_dot) / dir_normal_dot) * -1.0f;
			if (distance <= 0.0f) return false;
			Base::vec3 point = r.origin + r.dir * distance;
			if (!is_inside(t, point)) return false;
			return true;
		}

		RUN_ON_CPU_AND_GPU
		static Base::vec3 get_texcoord(const triangle& t, const Base::vec3& p)
		{
			double cap_area = length(cross(t.ca, p - t.a));
			double abp_area = length(cross(t.ab, p - t.b));
			double bcp_area = length(cross(t.bc, p - t.c));
			double u = cap_area / t.area;
			double v = abp_area / t.area;
			double w = bcp_area / t.area;
			return t.a_tex * u + t.b_tex * v + t.c_tex * w;
		}

		RUN_ON_CPU_AND_GPU
		static bool detect_hit(model& model, ray& ray, hit& hit_item, double& tnear)
		{
			double t0 = INFINITY;
			triangle* triangles = (triangle*)model.dshapes;
			bool hit = false;
			for (unsigned i = 0; i < model.shapes_size; i++)
			{
				if (does_intersect(triangles[i], ray, t0))
				{
					if (tnear > t0)
					{
						tnear = t0;
						hit_item.shape = &triangles[i];
						hit_item.pmodel = &model;
						hit = true;
					}
				}
			}
			return hit;
		}
	}
}
