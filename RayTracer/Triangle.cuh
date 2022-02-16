#pragma once
#include "Vector.cuh"

namespace RayTracer
{
	RUN_ON_GPU
	double edge_function(const Core::vec3& a, const Core::vec3& b, const Core::vec3& c)
	{
		return ((a.x - b.x) * (c.y - a.y)) - ((a.y - b.y) * (c.x - a.x));
	}

	RUN_ON_GPU
	bool is_inside(const triangle& t, const Core::vec3& p)
	{
		return (edge_function(t.c, t.a, p) < 0.0f) && (edge_function(p, t.a, t.b) < 0.0f) && (edge_function(t.c, p, t.b) < 0.0f);
	}

	RUN_ON_GPU
	bool does_intersect(const triangle& t, const Core::vec3& origin, const Core::vec3& dir, double& distance)
	{
		double dir_normal_dot = dot(dir, t.normal);
		if (dir_normal_dot >= 0.0f) return false;
		double origin_normal_dot = dot(origin, t.normal);
		distance = ((t.plane_distance + origin_normal_dot) / dir_normal_dot) * -1.0f;
		if (distance <= 0.0f) return false;
		Core::vec3 point = origin + dir * distance;
		if (!is_inside(t, point)) return false;
		return true;
	}

	RUN_ON_GPU
	Core::vec3 get_rgb(const triangle& t, const Core::vec3& p, const texture& tex)
	{
		double cap_area = length(cross(t.ca, p - t.a));
		double abp_area = length(cross(t.ab, p - t.b));
		double bcp_area = length(cross(t.bc, p - t.c));
		double u = cap_area / t.area;
		double v = abp_area / t.area;
		double w = bcp_area / t.area;
		Core::vec3 texcoords = t.a_tex * u + t.b_tex * v + t.c_tex * w;
		unsigned x = texcoords.x * tex.width;
		unsigned y = texcoords.y * tex.height;
		unsigned index = (y * tex.width + x) * 3.0;
		return Core::vec3{ tex.dtextures[index] / 255.0, tex.dtextures[index + 1] / 255.0, tex.dtextures[index + 2] / 255.0 };
	}
}
