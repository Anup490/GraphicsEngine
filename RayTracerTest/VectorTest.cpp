#include "pch.h"
#include "CppUnitTest.h"
#include "Vector.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace RayTracer;

namespace RayTracerTest
{
	TEST_CLASS(VectorTest)
	{
	public:
		
		bool equal(const Core::vec3& v1, const Core::vec3& v2)
		{
			return (v1.x == v2.x) && (v1.y == v2.y) && (v1.z == v2.z);
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
	};
}
