#include "pch.h"
#include "CppUnitTest.h"
#include "Vector.h"
#include "RayTracer.h"
#include <math.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace RayTracer;

namespace RayTracerTest
{
	TEST_CLASS(VectorTest)
	{
		bool equal(const Core::vec3& v1, const Core::vec3& v2)
		{
			return (v1.x == v2.x) && (v1.y == v2.y) && (v1.z == v2.z);
		}

		bool equal(const double& d1, const double& d2)
		{
			double diff = (d1 > d2) ? d1 - d2 : d2 - d1;
			return diff < 0.000001;
		}

		std::string to_string(const char* chars, const Core::vec3& v)
		{
			std::string s(chars);
			s.append(std::to_string(v.x));
			s.append(", ");
			s.append(std::to_string(v.y));
			s.append(", ");
			s.append(std::to_string(v.z));
			return s;
		}

		TEST_METHOD(VectorAddition)
		{
			Core::vec3 a{ 1, 2, 3 };
			Core::vec3 b{ 4, 5, 6 };
			Core::vec3 c{ 0, 0, 0 };
			c += a;
			c += b;
			Logger::WriteMessage(to_string("c = ", c).data());
			Assert::AreEqual(true, equal(a + b, c));
		}

		TEST_METHOD(VectorSubtraction)
		{
			Core::vec3 a{ 1, 2, 3 };
			Core::vec3 b{ 4, 5, 6 };
			Core::vec3 c(a);
			c -= b;
			Logger::WriteMessage(to_string("c = ", c).data());
			Assert::AreEqual(true, equal(a - b, c));
		}

		TEST_METHOD(VectorMultiplication)
		{
			Core::vec3 a{ 1, 2, 3 };
			Core::vec3 b{ 4, 5, 6 };
			Core::vec3 c{ 1, 1, 1 };
			c *= a;
			c *= b;
			Logger::WriteMessage(to_string("c = ", c).data());
			Assert::AreEqual(true, equal(a * b, c));
		}

		TEST_METHOD(VectorNormalization)
		{
			Core::vec3 a{ 5, 0, 0 };
			normalize(a);
			Logger::WriteMessage(to_string("a = ", a).data());
			Assert::AreEqual(true, length(a) == 1.0);
		}

		TEST_METHOD(VectorRotation)
		{
			double angle = 90;
			double angle_in_rad = (angle * 3.141592653589793) / 180.0;
			Core::vec3 v1{ 1, 0, 0 };
			Core::mat4 m;
			double matrix[16];
			matrix[0] = cos(angle_in_rad);
			matrix[1] = 0;
			matrix[2] = sin(angle_in_rad);
			matrix[3] = 0;
			matrix[4] = 0;
			matrix[5] = 1;
			matrix[6] = 0;
			matrix[7] = 0;
			matrix[8] = -sin(angle_in_rad);
			matrix[9] = 0;
			matrix[10] = cos(angle_in_rad);
			matrix[11] = 0;
			matrix[12] = 0;
			matrix[13] = 0;
			matrix[14] = 0;
			matrix[15] = 1;
			m.pmatrix = matrix;
			Core::vec3 v2 = m * v1;
			Logger::WriteMessage(to_string("v2 = ", v2).data());
			Assert::AreEqual(equal(v2.x, 0.0), true);
			Assert::AreEqual(equal(v2.y, 0.0), true);
			Assert::AreEqual(equal(v2.z, -1.0), true);
		}
	};
}
