#pragma once
#include "PrivateBase.cuh"

namespace RayTracer
{
	void draw_frame(RayTracer::pixels pixels, models models, double fov, Projection proj_type);
}
