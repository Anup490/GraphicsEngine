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
			double matrix[16];
			matrix[0] = 1;
			matrix[1] = 2;
			matrix[2] = 3;
			matrix[3] = 4;
			matrix[4] = 5;
			matrix[5] = 6;
			matrix[6] = 7;
			matrix[7] = 8;
			matrix[8] = 9;
			matrix[9] = 10;
			matrix[10] = 11;
			matrix[11] = 12;
			matrix[12] = 13;
			matrix[13] = 14;
			matrix[14] = 15;
			matrix[15] = 16;
			m.pmatrix = matrix;
			Core::vec3 v2 = m * v1;
			Assert::AreEqual(v2.x, 18.0);
			Assert::AreEqual(v2.y, 46.0);
			Assert::AreEqual(v2.z, 74.0);
		}

		TEST_METHOD(MatrixMatrixMultiplication)
		{
			Core::mat4 m1;
			double matrix1[16];
			matrix1[0] = 1;
			matrix1[1] = 2;
			matrix1[2] = 3;
			matrix1[3] = 4;
			matrix1[4] = 5;
			matrix1[5] = 6;
			matrix1[6] = 7;
			matrix1[7] = 8;
			matrix1[8] = 9;
			matrix1[9] = 10;
			matrix1[10] = 11;
			matrix1[11] = 12;
			matrix1[12] = 13;
			matrix1[13] = 14;
			matrix1[14] = 15;
			matrix1[15] = 16;
			m1.pmatrix = matrix1;
			Core::mat4 m2;
			double matrix2[16];
			matrix2[0] = 3;
			matrix2[1] = 4;
			matrix2[2] = 1;
			matrix2[3] = 2;
			matrix2[4] = 8;
			matrix2[5] = 3;
			matrix2[6] = 9;
			matrix2[7] = 7;
			matrix2[8] = 6;
			matrix2[9] = 7;
			matrix2[10] = 2;
			matrix2[11] = 4;
			matrix2[12] = 2;
			matrix2[13] = 8;
			matrix2[14] = 1;
			matrix2[15] = 9;
			m2.pmatrix = matrix2;
			Core::mat4 m = m1 * m2;
			Assert::AreEqual(m.pmatrix[0], 45.0);
			Assert::AreEqual(m.pmatrix[1], 63.0);
			Assert::AreEqual(m.pmatrix[2], 29.0);
			Assert::AreEqual(m.pmatrix[3], 64.0);
			Assert::AreEqual(m.pmatrix[4], 121.0);
			Assert::AreEqual(m.pmatrix[5], 151.0);
			Assert::AreEqual(m.pmatrix[6], 81.0);
			Assert::AreEqual(m.pmatrix[7], 152.0);
			Assert::AreEqual(m.pmatrix[8], 197.0);
			Assert::AreEqual(m.pmatrix[9], 239.0);
			Assert::AreEqual(m.pmatrix[10], 133.0);
			Assert::AreEqual(m.pmatrix[11], 240.0);
			Assert::AreEqual(m.pmatrix[12], 273.0);
			Assert::AreEqual(m.pmatrix[13], 327.0);
			Assert::AreEqual(m.pmatrix[14], 185.0);
			Assert::AreEqual(m.pmatrix[15], 328.0);
			delete[] m.pmatrix;
		}
	};
}