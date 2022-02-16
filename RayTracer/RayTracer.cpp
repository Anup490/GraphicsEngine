#include "pch.h"
#include "RayTracer.h"
#include "PrivateBase.cuh"
#include "Core.cuh"
#include "Vector.h"
#include <memory>
#include <vector>

namespace RayTracer
{
	model* dgpumodels = 0;
	unsigned models_size = 0;
	RayTracer::pixels* ppixels = 0;
	std::vector<triangle*>* p_all_triangles = 0;

	void prepare_data(const std::shared_ptr<std::vector<Core::model*>> pmodels, std::vector<model>& dmodels);
	void triangulate(const Core::model* pmodel, std::vector<RayTracer::triangle>* ptriangles);
	triangle make_triangle(Core::vertex a, Core::vertex b, Core::vertex c);
}

void RayTracer::init(std::shared_ptr<std::vector<Core::model*>> pmodels, int width, int height)
{
	if (((width % 32) != 0) || ((height % 32) != 0)) throw RayTraceException("width or height is not a multiple of 32");
	if (((width < 640) != 0) || ((height < 480) != 0)) throw RayTraceException("minimum acceptable resolution is 640x480");
	std::vector<model> dmodels;
	prepare_data(pmodels, dmodels);
	cudaMalloc(&dgpumodels, sizeof(model) * dmodels.size());
	cudaMemcpy(dgpumodels, dmodels.data(), sizeof(model) * dmodels.size(), cudaMemcpyHostToDevice);
	models_size = dmodels.size();
	Core::vec3* drgbs;
	cudaMalloc(&drgbs, sizeof(Core::vec3) * width * height);
	ppixels = new pixels(width, height);
	ppixels->data = drgbs;
}

std::unique_ptr<Core::vec3> RayTracer::render(double fov, Projection proj_type) throw(RayTraceException)
{
	if (!dgpumodels || !ppixels) throw RayTraceException("init function not called");
	draw_frame(*ppixels, models{ dgpumodels, models_size }, fov, proj_type);
	cudaDeviceSynchronize();
	Core::vec3* prgbs = new Core::vec3[ppixels->width * ppixels->height];
	cudaMemcpy(prgbs, ppixels->data, sizeof(Core::vec3) * ppixels->width * ppixels->height, cudaMemcpyDeviceToHost);
	return std::unique_ptr<Core::vec3>(prgbs);
}

void RayTracer::clear()
{
	for (triangle* dtriangle : *p_all_triangles)
	{
		cudaFree(dtriangle);
	}
	cudaFree(dgpumodels);
	cudaFree(ppixels->data);
	delete p_all_triangles;
	delete ppixels;
}

void RayTracer::prepare_data(const std::shared_ptr<std::vector<Core::model*>> pmodels, std::vector<model>& dmodels)
{
	p_all_triangles = new std::vector<triangle*>;
	for (unsigned i = 0; i < pmodels->size(); i++)
	{
		std::vector<triangle> triangles;
		triangulate((*pmodels)[i], &triangles);
		RayTracer::triangle* dgputriangles;
		cudaError_t stat;
		stat = cudaMalloc(&dgputriangles, sizeof(triangle) * triangles.size());
		stat = cudaMemcpy(dgputriangles, triangles.data(), sizeof(triangle) * triangles.size(), cudaMemcpyHostToDevice);
		RayTracer::model dmodel;
		dmodel.dtriangles = dgputriangles;
		dmodel.triangles_size = triangles.size();
		dmodels.push_back(dmodel);
		p_all_triangles->push_back(dgputriangles);
	}
}

void RayTracer::triangulate(const Core::model* pmodel, std::vector<RayTracer::triangle>* ptriangles)
{
	std::vector<Core::vertex>* pvertices = pmodel->pvertices;
	std::vector<unsigned>* pindices = pmodel->pindices;
	for (unsigned i = 0; i < pindices->size();)
	{
		Core::vertex a = (*pvertices)[(*pindices)[i++]];
		Core::vertex b = (*pvertices)[(*pindices)[i++]];
		Core::vertex c = (*pvertices)[(*pindices)[i++]];
		ptriangles->push_back(make_triangle(a, b, c));
	}
}

RayTracer::triangle RayTracer::make_triangle(Core::vertex a, Core::vertex b, Core::vertex c)
{
	Core::vec3 ab = b.position - a.position;
	Core::vec3 bc = c.position - b.position;
	Core::vec3 ca = a.position - c.position;
	Core::vec3 normal = cross(ab, bc);
	double area = length(normal);
	Core::vec3 emission{ 0.0, 0.0, 0.0 };
	normalize(normal);
	double plane_distance = dot(-normal, a.position);
	return triangle{ a.position,b.position,c.position,ab,bc,ca,a.texcoord,b.texcoord,c.texcoord,emission, normal,plane_distance,area };
}