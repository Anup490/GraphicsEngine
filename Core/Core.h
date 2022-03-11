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
		unsigned char* ptextures = 0;
		int width = 0;
		int height = 0;
		int channels = 0;
	};

	struct model
	{
		Core::vec3 position, emissive_color;
		std::vector<vertex>* pvertices = 0;
		std::vector<unsigned>* pindices = 0;
		texture diffuse;
		texture specular;
		double reflectivity = 0.0, transparency = 0.0;
		~model() { if (pvertices) delete pvertices; if(pindices) delete pindices; }
	};

	struct triangle
	{
		vertex a, b, c;
	};
}