#include "pch.h"
#include "Core.h"
#include "math.h"

using namespace Core;

vec3::vec3() :x(0.0f), y(0.0f), z(0.0f), length(sqrt(this->dot(*this))) {}
vec3::vec3(double f) : x(f), y(f), z(f), length(sqrt(this->dot(*this))) {}
vec3::vec3(double x, double y, double z) : x(x), y(y), z(z), length(sqrt(this->dot(*this))) {}
double vec3::dot(const vec3& v) const { return x * v.x + y * v.y + z * v.z; }
vec3 vec3::cross(const vec3& v) const { return vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }
void vec3::normalize() { *this = vec3(x / length, y / length, z / length); }
vec3 vec3::operator*(const double& f) const { return vec3(x * f, y * f, z * f); }
vec3 vec3::operator+(const vec3& v) const { return vec3(x + v.x, y + v.y, z + v.z); }
vec3 vec3::operator*(const vec3& v) const { return vec3(x * v.x, y * v.y, z * v.z); }
vec3 vec3::operator-(const vec3& v) const { return vec3(x - v.x, y - v.y, z - v.z); }
vec3 vec3::operator-() const { return vec3(-x, -y, -z); }
vec3& vec3::operator+=(const vec3& v) { *this = *this + v; return *this; }
vec3& vec3::operator*=(const vec3& v) { *this = *this * v; return *this; }
vec3& vec3::operator-=(const vec3& v) { *this = *this - v; return *this; }