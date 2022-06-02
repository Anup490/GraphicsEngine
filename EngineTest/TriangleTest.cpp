#include "pch.h"
#include "CppUnitTest.h"
#include "Triangle.cuh"

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
			s.append("\n");
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

		TEST_METHOD(EdgeFunctionTest)
		{
			Base::vec3 a{ 1, 1 };
			Base::vec3 b{ 5, 2 };
			Base::vec3 c{ 3, 4 };
			Base::vec3 p{ 3, 2 };
			Assert::AreEqual(true, Triangle::is_inside(triangle{ a,b,c }, p));
		}

		TEST_METHOD(TesellationTest)
		{
			Base::vec3 a{ 4, 100 };
			Base::vec3 b{ 0, 0 };
			Base::vec3 c{ 300, 600 };
			double n = 4;
			double n_half = n / 2;
			double inc_c = 1, inc_a = 1;
			int it = 2 * n - 2;
			Base::vec3 t_a, t_b, t_c;
			bool swap = false;
			t_a = a * ((n_half - inc_a) / n_half) + b * (inc_a / n_half);
			t_b = b;
			t_c = b * ((n - inc_c) / n) + c * (inc_c / n);
			inc_a++;
			inc_c++;
			Logger::WriteMessage("Triangle\n");
			Logger::WriteMessage(to_string(t_a).c_str());
			Logger::WriteMessage(to_string(t_b).c_str());
			Logger::WriteMessage(to_string(t_c).c_str());
			Logger::WriteMessage("\n\n");
			for (int i = 1; i < it; i++)
			{
				if (i > (it/2))
				{
					if (i % 2 == 0)
					{
						t_b = t_a;
						t_a = a * (inc_a / n_half) + c * ((n_half - inc_a) / n_half);
						inc_a++;
						if (inc_a > n_half) inc_a = 1;

						Logger::WriteMessage("Triangle\n");
						Logger::WriteMessage(to_string(t_a).c_str());
						Logger::WriteMessage(to_string(t_b).c_str());
						Logger::WriteMessage(to_string(t_c).c_str());
						Logger::WriteMessage("\n\n");
					}
					else
					{
						t_b = t_c;
						t_c = b * ((n - inc_c) / n) + c * (inc_c / n);
						inc_c++;

						Logger::WriteMessage("Triangle\n");
						Logger::WriteMessage(to_string(t_a).c_str());
						Logger::WriteMessage(to_string(t_b).c_str());
						Logger::WriteMessage(to_string(t_c).c_str());
						Logger::WriteMessage("\n\n");
					}
				}
				else
				{
					if (i % 2 == 0)
					{
						t_b = t_a;
						t_a = a * (inc_a / n_half) + c * ((n_half - inc_a) / n_half);
						inc_a++;
						if (inc_a > n_half) inc_a = 1;

						Logger::WriteMessage("Triangle\n");
						Logger::WriteMessage(to_string(t_a).c_str());
						Logger::WriteMessage(to_string(t_b).c_str());
						Logger::WriteMessage(to_string(t_c).c_str());
						Logger::WriteMessage("\n\n");
					}
					else
					{
						t_b = t_c;
						t_c = b * ((n - inc_c) / n) + c * (inc_c / n);
						inc_c++;

						Logger::WriteMessage("Triangle\n");
						Logger::WriteMessage(to_string(t_a).c_str());
						Logger::WriteMessage(to_string(t_b).c_str());
						Logger::WriteMessage(to_string(t_c).c_str());
						Logger::WriteMessage("\n\n");
					}
				}
				
			}
		}
	};
}