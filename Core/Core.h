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
	enum class shape_type { TRIANGLE, SPHERE, BOX };
	enum class model_type { LIGHT, CAMERA, OBJECT };

	struct vec3
	{
		double x = 0.0, y = 0.0, z = 0.0;
	};

	struct mat4
	{
		double* pmatrix;
		const unsigned size = 16;
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

	struct cubemap
	{
		texture left;
		texture right;
		texture bottom;
		texture top;
		texture front;
		texture back;
	};

	struct model
	{
		vec3 position, emissive_color, surface_color, front{ 0,0,1 }, right{ 1,0,0 }, up{ 0,1,0 };
		void* pshapes = 0;
		unsigned shapes_size = 0;
		texture diffuse;
		texture specular;
		double reflectivity = 0.0, transparency = 0.0;
		shape_type s_type;
		model_type m_type;
		~model() { if (pshapes) delete[] pshapes; }
	};

	struct model_info
	{
		const char* file_path;
		Core::vec3 position;
	};

	struct triangle
	{
		vertex a, b, c;
	};

	struct sphere
	{
		double radius = 0.0;
		Core::vec3 center;
	};

	struct box
	{
		Core::vec3 min, max, center{ (min.x + max.x)/2.0, (min.y + max.y) / 2.0, (min.z + max.z) / 2.0 };
	};
}