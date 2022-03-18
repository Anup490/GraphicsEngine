#include "PrivateBase.cuh"

namespace RayTracer
{
	RUN_ON_CPU_AND_GPU
	Core::vec3 operator*(const Core::mat4& m, const Core::vec3& v)
	{
		double x = m.matrix[0] * v.x + m.matrix[1] * v.y + m.matrix[2] * v.z;
		double y = m.matrix[3] * v.x + m.matrix[4] * v.y + m.matrix[5] * v.z;
		double z = m.matrix[6] * v.x + m.matrix[7] * v.y + m.matrix[8] * v.z;
		return Core::vec3{ x, y, z };
	}

	RUN_ON_CPU_AND_GPU
	Core::mat4 operator*(const Core::mat4& m1, const Core::mat4& m2)
	{
		Core::mat4 m;
		for (unsigned r=0; r<16; r+=4)
		{
			for (unsigned c=0; c<4; c++)
			{
				m.matrix[c+r] = m1.matrix[r] * m2.matrix[c] + m1.matrix[r+1] * m2.matrix[c+4] + m1.matrix[r+2] * m2.matrix[c+8] + m1.matrix[r + 3] * m2.matrix[c + 12];
			}
		}
		return m;
	}
}