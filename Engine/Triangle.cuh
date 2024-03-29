#pragma once
#include "EngineCore.cuh"
#include "Vector.cuh"

namespace Engine
{
	namespace Triangle
	{
		RUN_ON_GPU
		static double edge_function(const Base::vec3& a, const Base::vec3& b, const Base::vec3& c)
		{
			return ((a.x - b.x) * (c.y - a.y)) - ((a.y - b.y) * (c.x - a.x));
		}

		RUN_ON_GPU
		static bool is_inside(const triangle& t, const Base::vec3& p)
		{
			return (edge_function(t.c, t.a, p) <= 0.0) && (edge_function(p, t.a, t.b) <= 0.0) && (edge_function(t.c, p, t.b) <= 0.0);
		}

		RUN_ON_GPU
		static void make_triangle(const Base::vec3& a, const Base::vec3& b, const Base::vec3& c, triangle& t)
		{
			t.ab = b - a;
			t.bc = c - b;
			t.ca = a - c;
			t.normal = cross(t.ab, t.bc);
			t.area = length(t.normal) / 2.0;
			normalize(t.normal);
			t.plane_distance = dot(-t.normal, a);
			t.a = a;
			t.b = b;
			t.c = c;
		}

		RUN_ON_GPU
		static Base::vec3 interpolate_point(const triangle& t_raster, const triangle& t_view, const Base::vec3& raster_coord)
		{
			double cap_area = length(cross(t_raster.ca, raster_coord - t_raster.a)) / 2.0;
			double abp_area = length(cross(t_raster.ab, raster_coord - t_raster.b)) / 2.0;
			double bcp_area = length(cross(t_raster.bc, raster_coord - t_raster.c)) / 2.0;
			double u = cap_area / t_raster.area;
			double v = abp_area / t_raster.area;
			double w = bcp_area / t_raster.area;
			Base::vec3 point;
			point.x = t_view.a.x * w + t_view.b.x * u + t_view.c.x * v;
			point.y = t_view.a.y * w + t_view.b.y * u + t_view.c.y * v;
			double invz = (w / t_view.a.z) + (u / t_view.b.z) + (v / t_view.c.z);
			double z = 1 / invz;
			point.z = (z < 0.0) ? (z * -1.0) : z;
			return point;
		}

		RUN_ON_GPU
		static Base::vec3 get_interpolated_texcoord(const triangle& t_view, const triangle* ptriangle, const Base::vec3 uvw, const double& depth)
		{
			Base::vec3 texcoord;
			texcoord.x = depth * (((uvw.z * ptriangle->a_tex.x) / t_view.a.z) + ((uvw.x * ptriangle->b_tex.x) / t_view.b.z) + ((uvw.y * ptriangle->c_tex.x) / t_view.c.z));
			texcoord.y = depth * (((uvw.z * ptriangle->a_tex.y) / t_view.a.z) + ((uvw.x * ptriangle->b_tex.y) / t_view.b.z) + ((uvw.y * ptriangle->c_tex.y) / t_view.c.z));
			return texcoord;
		}

		RUN_ON_GPU
		static Base::vec3 interpolate_texcoord(const triangle& t_raster, const triangle& t_view, const triangle* ptriangle, const Base::vec3& raster_coord, const double& depth)
		{
			double cap_area = length(cross(t_raster.ca, raster_coord - t_raster.a)) / 2.0;
			double abp_area = length(cross(t_raster.ab, raster_coord - t_raster.b)) / 2.0;
			double bcp_area = length(cross(t_raster.bc, raster_coord - t_raster.c)) / 2.0;
			double u = cap_area / t_raster.area;
			double v = abp_area / t_raster.area;
			double w = bcp_area / t_raster.area;
			return get_interpolated_texcoord(t_view, ptriangle, Base::vec3{ u,v,w }, depth);
		}

		RUN_ON_GPU
		static bool does_intersect(const triangle& t, const ray& r, double& distance)
		{
			double dir_normal_dot = dot(r.dir, t.normal);
			if (dir_normal_dot >= 0.0) return false;
			double origin_normal_dot = dot(r.origin, t.normal);
			distance = ((t.plane_distance + origin_normal_dot) / dir_normal_dot) * -1.0;
			if (distance <= 0.0) return false;
			Base::vec3 point = r.origin + r.dir * distance;
			if (!is_inside(t, point)) return false;
			return true;
		}

		RUN_ON_GPU
		static Base::vec3 get_texcoord(const triangle& t, const Base::vec3& p)
		{
			double cap_area = length(cross(t.ca, p - t.a)) / 2.0;
			double abp_area = length(cross(t.ab, p - t.b)) / 2.0;
			double bcp_area = length(cross(t.bc, p - t.c)) / 2.0;
			double u = cap_area / t.area;
			double v = abp_area / t.area;
			double w = bcp_area / t.area;
			return t.a_tex * w + t.b_tex * u + t.c_tex * v;
		}

		RUN_ON_GPU
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
