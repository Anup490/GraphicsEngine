#include "pch.h"
#include "CppUnitTest.h"
#include "Vector.cuh"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace RayTracer;

namespace RayTracerTest
{
	TEST_CLASS(RayTracerTest)
	{
	public:
		
		bool equal(const vec3& v1, const vec3& v2)
		{
			return (v1.x == v2.x) && (v1.y == v2.y) && (v1.z == v2.z);
		}

		std::string to_string(const char* chars, const vec3& v)
		{
			std::string s(chars);
			s.append(std::to_string(v.x));
			s.append(", ");
			s.append(std::to_string(v.y));
			s.append(", ");
			s.append(std::to_string(v.z));
			s.append(" || ");
			return s;
		}

		TEST_METHOD(VectorTest)
		{
			vec3 a(1, 2, 3);
			vec3 b(4, 5, 6);
			vec3 d(0, 0, 0);
			vec3 e(1, 1, 1);
			vec3 f(a);
			d += a;
			d += b;
			e *= a;
			e *= b;
			f -= b;
			Logger::WriteMessage(to_string("d = ", d).data());
			Assert::AreEqual(true, equal(a + b, d));
			Logger::WriteMessage(to_string("e = ", e).data());
			Assert::AreEqual(true, equal(a * b, e));
			Logger::WriteMessage(to_string("f = ", f).data());
			Assert::AreEqual(true, equal(a - b, f));
		}
	};
}
