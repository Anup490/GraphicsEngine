#include "Core.h"

namespace RayTracer
{
	Core::vec3 operator*(const Core::mat3& m, const Core::vec3& v)
	{
		double x = m.matrix[0] * v.x + m.matrix[1] * v.y + m.matrix[2] * v.z;
		double y = m.matrix[3] * v.x + m.matrix[4] * v.y + m.matrix[5] * v.z;
		double z = m.matrix[6] * v.x + m.matrix[7] * v.y + m.matrix[8] * v.z;
		return Core::vec3{ x, y, z };
	}

	Core::mat3 operator*(const Core::mat3& m1, const Core::mat3& m2)
	{
		Core::mat3 m;
		for (unsigned r=0; r<9; r+=3)
		{
			for (unsigned c=0; c<3; c++)
			{
				m.matrix[c+r] = m1.matrix[r] * m2.matrix[c] + m1.matrix[r+1] * m2.matrix[c+3] + m1.matrix[r+2] * m2.matrix[c+6];
			}
		}
		return m;
	}
}