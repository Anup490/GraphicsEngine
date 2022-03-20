#include "pch.h"
#include "CppUnitTest.h"
#include "RayTracer.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace RayTracer;

namespace RayTracerTest
{
	TEST_CLASS(MatrixTest)
	{
		TEST_METHOD(MatrixVectorMultiplication)
		{
			Core::vec3 v1{ 1, 2, 3 };
			Core::mat4 m;
			m.matrix[0] = 1;
			m.matrix[1] = 2;
			m.matrix[2] = 3;
			m.matrix[3] = 4;
			m.matrix[4] = 5;
			m.matrix[5] = 6;
			m.matrix[6] = 7;
			m.matrix[7] = 8;
			m.matrix[8] = 9;
			m.matrix[9] = 10;
			m.matrix[10] = 11;
			m.matrix[11] = 12;
			m.matrix[12] = 13;
			m.matrix[13] = 14;
			m.matrix[14] = 15;
			m.matrix[15] = 16;
			Core::vec3 v2 = m * v1;
			Assert::AreEqual(v2.x, 18.0);
			Assert::AreEqual(v2.y, 46.0);
			Assert::AreEqual(v2.z, 74.0);
		}

		TEST_METHOD(MatrixMatrixMultiplication)
		{
			Core::mat4 m1;
			m1.matrix[0] = 1;
			m1.matrix[1] = 2;
			m1.matrix[2] = 3;
			m1.matrix[3] = 4;
			m1.matrix[4] = 5;
			m1.matrix[5] = 6;
			m1.matrix[6] = 7;
			m1.matrix[7] = 8;
			m1.matrix[8] = 9;
			m1.matrix[9] = 10;
			m1.matrix[10] = 11;
			m1.matrix[11] = 12;
			m1.matrix[12] = 13;
			m1.matrix[13] = 14;
			m1.matrix[14] = 15;
			m1.matrix[15] = 16;
			Core::mat4 m2;
			m2.matrix[0] = 3;
			m2.matrix[1] = 4;
			m2.matrix[2] = 1;
			m2.matrix[3] = 2;
			m2.matrix[4] = 8;
			m2.matrix[5] = 3;
			m2.matrix[6] = 9;
			m2.matrix[7] = 7;
			m2.matrix[8] = 6;
			m2.matrix[9] = 7;
			m2.matrix[10] = 2;
			m2.matrix[11] = 4;
			m2.matrix[12] = 2;
			m2.matrix[13] = 8;
			m2.matrix[14] = 1;
			m2.matrix[15] = 9;
			Core::mat4 m = m1 * m2;
			Assert::AreEqual(m.matrix[0], 45.0);
			Assert::AreEqual(m.matrix[1], 63.0);
			Assert::AreEqual(m.matrix[2], 29.0);
			Assert::AreEqual(m.matrix[3], 64.0);
			Assert::AreEqual(m.matrix[4], 121.0);
			Assert::AreEqual(m.matrix[5], 151.0);
			Assert::AreEqual(m.matrix[6], 81.0);
			Assert::AreEqual(m.matrix[7], 152.0);
			Assert::AreEqual(m.matrix[8], 197.0);
			Assert::AreEqual(m.matrix[9], 239.0);
			Assert::AreEqual(m.matrix[10], 133.0);
			Assert::AreEqual(m.matrix[11], 240.0);
			Assert::AreEqual(m.matrix[12], 273.0);
			Assert::AreEqual(m.matrix[13], 327.0);
			Assert::AreEqual(m.matrix[14], 185.0);
			Assert::AreEqual(m.matrix[15], 328.0);
		}
	};
}