#include "EngineCore.cuh"

namespace Engine
{
	RUN_ON_CPU_AND_GPU
	Base::vec3 operator*(const Base::mat4& m, const Base::vec3& v)
	{
		double x = m.pmatrix[0] * v.x + m.pmatrix[1] * v.y + m.pmatrix[2] * v.z + m.pmatrix[3];
		double y = m.pmatrix[4] * v.x + m.pmatrix[5] * v.y + m.pmatrix[6] * v.z + m.pmatrix[7];
		double z = m.pmatrix[8] * v.x + m.pmatrix[9] * v.y + m.pmatrix[10] * v.z + m.pmatrix[11];
		double w = m.pmatrix[12] * v.x + m.pmatrix[13] * v.y + m.pmatrix[14] * v.z + m.pmatrix[15];
		return Base::vec3{ x/w, y/w, z/w };
	}

	RUN_ON_CPU_AND_GPU
	Base::mat4 operator*(const Base::mat4& m1, const Base::mat4& m2)
	{
		Base::mat4 m;
		m.pmatrix = new double[16];
		for (unsigned r=0; r<16; r+=4)
		{
			for (unsigned c=0; c<4; c++)
			{
				m.pmatrix[c+r] = m1.pmatrix[r] * m2.pmatrix[c] + m1.pmatrix[r+1] * m2.pmatrix[c+4] + m1.pmatrix[r+2] * m2.pmatrix[c+8] + m1.pmatrix[r + 3] * m2.pmatrix[c + 12];
			}
		}
		return m;
	}
}