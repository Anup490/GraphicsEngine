#include "PrivateBase.cuh"
#include "Maths.h"
#include "Vector.h"

namespace RayTracer
{
	namespace Box
	{
		RUN_ON_GPU
		face get_face(const Base::box* pbox, const Base::vec3& phit)
		{
			if (equal(phit.x, pbox->max.x)) return face::LEFT;
			if (equal(phit.x, pbox->min.x)) return face::RIGHT;
			if (equal(phit.y, pbox->max.y)) return face::TOP;
			if (equal(phit.y, pbox->min.y)) return face::BOTTOM;
			if (equal(phit.z, pbox->max.z)) return face::FRONT;
			if (equal(phit.z, pbox->min.z)) return face::BACK;
			return face::NONE;
		}

		RUN_ON_GPU
		Base::vec3 calculate_normal(const Base::box* pbox, const Base::vec3& phit)
		{
			face f = get_face(pbox, phit);
			Base::vec3 inside{};
			if ((f == face::LEFT) || (f == face::RIGHT)) inside = Base::vec3{ pbox->center.x, phit.y, phit.z };
			if ((f == face::TOP) || (f == face::BOTTOM)) inside = Base::vec3{ phit.x, pbox->center.y, phit.z };
			if ((f == face::FRONT) || (f == face::BACK)) inside = Base::vec3{ phit.x, phit.y, pbox->center.z };
			return phit - inside;
		}

		RUN_ON_GPU
		bool does_intersect(const Base::box& b, const ray& r, double& distance)
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
			Base::box* boxes = (Base::box*)model.dshapes;
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