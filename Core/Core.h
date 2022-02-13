#pragma once
#include <math.h>

namespace std
{
	template <class _Ty>
	class allocator;

	template <class _Ty, class _Alloc = allocator<_Ty>>
	class vector;
}

namespace Core
{
	struct vec3
	{
		double x, y, z;
		double length;
		vec3() : x(0.0f), y(0.0f), z(0.0f), length(sqrt(0.0)) {}
		vec3(double f) : x(f), y(f), z(f), length(sqrt(3.0 * f * f)) {}
		vec3(double x, double y, double z) : x(x), y(y), z(z), length(sqrt(x* x + y * y + z * z)) {}
	};

	struct vertex
	{
		vec3 position, normal, texcoord;
	};

	struct triangle
	{
		vertex a, b, c;
	};

	struct model
	{
		std::vector<vertex>* pvertices = 0;
		std::vector<float>* ptextures = 0;
		std::vector<unsigned>* pindices = 0;
		~model() { if (pvertices) delete pvertices; if (ptextures) delete ptextures; if(pindices) delete pindices; }
	};
}