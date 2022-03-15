#include "pch.h"
#include "CppUnitTest.h"
#include "Matrix.cuh"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace RayTracer;

namespace RayTracerTest
{
	TEST_CLASS(MatrixTest)
	{
		TEST_METHOD(MatrixVectorMultiplication)
		{
			Core::vec3 v1{ 1, 2, 3 };
			Core::mat3 m;
			m.matrix[0] = 1;
			m.matrix[1] = 2;
			m.matrix[2] = 3;
			m.matrix[3] = 4;
			m.matrix[4] = 5;
			m.matrix[5] = 6;
			m.matrix[6] = 7;
			m.matrix[7] = 8;
			m.matrix[8] = 9;
			Core::vec3 v2 = m * v1;
			Assert::AreEqual(v2.x, 14.0);
			Assert::AreEqual(v2.y, 32.0);
			Assert::AreEqual(v2.z, 50.0);
		}

		TEST_METHOD(MatrixMatrixMultiplication)
		{
			Core::mat3 m1;
			m1.matrix[0] = 1;
			m1.matrix[1] = 2;
			m1.matrix[2] = 3;
			m1.matrix[3] = 4;
			m1.matrix[4] = 5;
			m1.matrix[5] = 6;
			m1.matrix[6] = 7;
			m1.matrix[7] = 8;
			m1.matrix[8] = 9;
			Core::mat3 m2;
			m2.matrix[0] = 3;
			m2.matrix[1] = 4;
			m2.matrix[2] = 1;
			m2.matrix[3] = 2;
			m2.matrix[4] = 8;
			m2.matrix[5] = 3;
			m2.matrix[6] = 9;
			m2.matrix[7] = 7;
			m2.matrix[8] = 6;
			Core::mat3 m = m1 * m2;
			Assert::AreEqual(m.matrix[0], 34.0);
			Assert::AreEqual(m.matrix[1], 41.0);
			Assert::AreEqual(m.matrix[2], 25.0);
			Assert::AreEqual(m.matrix[3], 76.0);
			Assert::AreEqual(m.matrix[4], 98.0);
			Assert::AreEqual(m.matrix[5], 55.0);
			Assert::AreEqual(m.matrix[6], 118.0);
			Assert::AreEqual(m.matrix[7], 155.0);
			Assert::AreEqual(m.matrix[8], 85.0);
		}
	};
}