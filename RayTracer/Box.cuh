#include "PrivateBase.cuh"
#include "Maths.h"
#include "Vector.h"

namespace RayTracer
{
	namespace Box
	{
		enum class face { LEFT, RIGHT, BOTTOM, TOP, FRONT, BACK };

		RUN_ON_GPU
		face get_face(Core::box* pbox, Core::vec3& phit)
		{
			if (pbox->min.x == phit.x) return face::LEFT;
			else if (pbox->max.x == phit.x) return face::RIGHT;
			else if (pbox->min.y == phit.y) return face::BOTTOM;
			else if (pbox->max.y == phit.y) return face::TOP;
			else if (pbox->min.z == phit.z) return face::FRONT;
			else return face::BACK;
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