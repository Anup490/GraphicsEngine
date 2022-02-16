#pragma once

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
		double x = 0.0, y = 0.0, z = 0.0;
	};

	struct vertex
	{
		vec3 position, normal, texcoord;
	};

	struct texture
	{
		std::vector<float>* ptextures = 0;
		int width = 0;
		int height = 0;
		int channels = 0;
		~texture() { if (ptextures) delete ptextures; }
	};

	struct model
	{
		Core::vec3 position;
		std::vector<vertex>* pvertices = 0;
		std::vector<unsigned>* pindices = 0;
		texture texture_data;
		~model() { if (pvertices) delete pvertices; if(pindices) delete pindices; }
	};

	struct triangle
	{
		vertex a, b, c;
	};
}