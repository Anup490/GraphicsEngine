#include "pch.h"
#include "CppUnitTest.h"
#include "Vector.h"
#include "Engine.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Engine;

namespace EngineTest
{
	TEST_CLASS(TriangleTest)
	{
		Engine::triangle make_triangle(Base::vertex& a, Base::vertex& b, Base::vertex& c)
		{
			Base::vec3 ab = b.position - a.position;
			Base::vec3 bc = c.position - b.position;
			Base::vec3 ca = a.position - c.position;
			Base::vec3 normal = cross(ab, bc);
			double area = length(normal);
			Base::vec3 emission{ 0.0, 0.0, 0.0 };
			normalize(normal);
			double plane_distance = dot(-normal, a.position);
			return triangle{ a.position,b.position,c.position,ab,bc,ca,a.texcoord,b.texcoord,c.texcoord,normal,plane_distance,area };
		}

		std::string to_string(const Base::vec3& v)
		{
			std::string s("Value :: ( ");
			s.append(std::to_string(v.x));
			s.append(", ");
			s.append(std::to_string(v.y));
			s.append(", ");
			s.append(std::to_string(v.z));
			s.append(" )");
			return s;
		}

		std::string to_string(const Base::vertex& v)
		{
			Base::vec3 position = v.position;
			Base::vec3 normal = v.normal;
			Base::vec3 texcoord = v.texcoord;
			std::string s("Vertex := position( ");
			s.append(std::to_string(position.x));
			s.append(", ");
			s.append(std::to_string(position.y));
			s.append(", ");
			s.append(std::to_string(position.z));
			s.append(" ) || normal( ");
			s.append(std::to_string(normal.x));
			s.append(", ");
			s.append(std::to_string(normal.y));
			s.append(", ");
			s.append(std::to_string(normal.z));
			s.append(" ) || texcoord( ");
			s.append(std::to_string(texcoord.x));
			s.append(", ");
			s.append(std::to_string(texcoord.y));
			s.append(" )");
			return s;
		}

		TEST_METHOD(BarycentricTest)
		{
			Base::vec3 a_pos{ 2.0, 1.0, -4.0 };
			Base::vec3 b_pos{ 8.0, 4.0, -4.0 };
			Base::vec3 c_pos{ 5.0, 8.0, -4.0 };
			Base::vec3 a_normal;
			Base::vec3 b_normal;
			Base::vec3 c_normal;
			Base::vec3 a_tex{ 0.0, 0.0 };
			Base::vec3 b_tex{ 1.0, 0.0 };
			Base::vec3 c_tex{ 1.0, 1.0 };
			Base::vertex a{ a_pos, a_normal, a_tex };
			Base::vertex b{ b_pos, b_normal, b_tex };
			Base::vertex c{ c_pos, c_normal, c_tex };
			Engine::triangle triangle = make_triangle(a, b, c);
			Base::vec3 origin;
			Base::vec3 dir{ 5.0, 4.0, -4.0 };
			normalize(dir);
			double t = 0.0;
			Triangle::does_intersect(triangle, ray{ origin, dir }, t);
			Base::vec3 v = origin + (dir * t);
			Logger::WriteMessage(to_string(Triangle::get_texcoord(triangle, v)).data());
		}
	};
}