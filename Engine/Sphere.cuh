#pragma once
#include "EngineCore.cuh"
#include "Vector.cuh"
#include "Maths.cuh"

namespace Engine
{
	namespace Sphere
	{
		RUN_ON_GPU
		static bool does_intersect(const sphere& s, const ray& r, double& distance)
		{
			Base::vec3 l = s.center - r.origin;
			double tca = dot(l, r.dir);
			double d_square = dot(l, l) - (tca * tca);
			if (d_square > s.radius_square) return false;
			double thc = square_root(s.radius_square - d_square);
			double tfirst = tca - thc;
			double tlast = tca + thc;
			if ((tfirst < 0.0) && (tlast < 0.0)) return false;
			distance = (tfirst < 0.0) ? tlast : tfirst;
			return true;
		}

		RUN_ON_GPU
		static bool detect_hit(model& model, ray& ray, hit& hit_item, double& tnear)
		{
			double t0 = get_infinity();
			sphere* spheres = (sphere*)model.dshapes;
			bool hit = false;
			for (unsigned i = 0; i < model.shapes_size; i++)
			{
				if (does_intersect(spheres[i], ray, t0))
				{
					if (tnear > t0)
					{
						tnear = t0;
						hit_item.shape = &spheres[i];
						hit_item.pmodel = &model;
						hit = true;
					}
				}
			}
			return hit;
		}
	}
}