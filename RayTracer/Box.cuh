#include "PrivateBase.cuh"
#include "Maths.h"
#include "Vector.h"

namespace RayTracer
{
	namespace Box
	{
		RUN_ON_GPU
		face get_face(const Core::box* pbox, const Core::vec3& phit)
		{
			if (pbox->max.x == phit.x) return face::LEFT;
			else if (pbox->min.x == phit.x) return face::RIGHT;
			else if (pbox->max.y == phit.y) return face::TOP;
			else if (pbox->min.y == phit.y) return face::BOTTOM;
			else if (pbox->max.z == phit.z) return face::BACK;
			else if (pbox->min.z == phit.z) return face::FRONT;
			return face::NONE;
		}

		RUN_ON_GPU
		Core::vec3 calculate_normal(const Core::box* pbox, const Core::vec3& phit)
		{
			face f = get_face(pbox, phit);
			if (f == face::LEFT) return Core::vec3{ pbox->max.x, pbox->center.y, pbox->center.z } - pbox->center;
			if (f == face::RIGHT) return Core::vec3{ pbox->min.x, pbox->center.y, pbox->center.z } - pbox->center;
			if (f == face::BOTTOM) return Core::vec3{ pbox->center.x, pbox->min.y, pbox->center.z } - pbox->center;
			if (f == face::TOP) return Core::vec3{ pbox->center.x, pbox->max.y, pbox->center.z } - pbox->center;
			if (f == face::FRONT) return Core::vec3{ pbox->center.x, pbox->center.y, pbox->max.z } - pbox->center;
			if (f == face::BACK) Core::vec3{ pbox->center.x, pbox->center.y, pbox->min.z } - pbox->center;
			return Core::vec3{};
		}

		RUN_ON_GPU
		bool does_intersect(const Core::box& b, const ray& r, double& distance)
		{
			double txmin = (b.min.x - r.origin.x) / r.dir.x;
			double txmax = (b.max.x - r.origin.x) / r.dir.x;
			double tymin = (b.min.y - r.origin.y) / r.dir.y;
			double tymax = (b.max.y - r.origin.y) / r.dir.y;
			double tzmin = (b.min.z - r.origin.z) / r.dir.z;
			double tzmax = (b.max.z - r.origin.z) / r.dir.z;
			if (txmin > txmax) swap(txmin, txmax);
			if (tymin > tymax) swap(tymin, tymax);
			if (tzmin > tzmax) swap(tzmin, tzmax);
			if (txmin > tymax) return false;
			if (tymin > txmax) return false;
			if (tzmin > txmax) return false;
			if (txmin > tzmax) return false;
			if (tymin > tzmax) return false;
			if (tzmin > tymax) return false;
			double tfirst = maximum(txmin, tymin, tzmin);
			double tlast = minimum(txmax, tymax, tzmax);
			if ((tfirst < 0.0) && (tlast < 0.0)) return false;
			distance = (tfirst < 0.0) ? tlast : tfirst;
			return true;
		}

		RUN_ON_GPU
		bool detect_hit(model& model, ray& ray, hit& hit_item, double& tnear)
		{
			double t0 = get_infinity();
			Core::box* boxes = (Core::box*)model.dshapes;
			bool hit = false;
			for (unsigned i = 0; i < model.shapes_size; i++)
			{
				if (does_intersect(boxes[i], ray, t0))
				{
					if (tnear > t0)
					{
						tnear = t0;
						hit_item.shape = &boxes[i];
						hit_item.pmodel = &model;
						hit = true;
					}
				}
			}
			return hit;
		}
	}
}