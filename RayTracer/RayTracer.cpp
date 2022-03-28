#include "pch.h"
#include "RayTracer.h"
#include "PrivateBase.cuh"
#include "Core.cuh"
#include "Vector.h"
#include <memory>
#include <vector>

namespace RayTracer
{
	world* dworld = 0;
	model* dgpumodels = 0;
	unsigned models_size = 0;
	RayTracer::pixels* ppixels = 0;
	std::vector<void*>* p_all_shapes = 0;
	std::vector<unsigned char*>* p_all_textures = 0;

	void prepare_data(const std::shared_ptr<std::vector<Core::model*>> pmodels, std::vector<model>& dmodels);
	RayTracer::texture get_texture(Core::texture core_texture);
	void prepare_triangles(const Core::model* pmodel, std::vector<RayTracer::triangle>* ptriangles);
	void prepare_spheres(const Core::model* pmodel, std::vector<RayTracer::sphere>* pspheres);
	triangle make_triangle(Core::vertex a, Core::vertex b, Core::vertex c);
	input* prepare_inputs(input& i);
}

void RayTracer::init(std::shared_ptr<std::vector<Core::model*>> pmodels, int width, int height)
{
	if (((width % 32) != 0) || ((height % 32) != 0)) throw RayTraceException("width or height is not a multiple of 32");
	if (((width < 640) != 0) || ((height < 480) != 0)) throw RayTraceException("minimum acceptable resolution is 640x480");
	std::vector<model> dmodels;
	p_all_shapes = new std::vector<void*>;
	p_all_textures = new std::vector<unsigned char*>();
	prepare_data(pmodels, dmodels);
	cudaMalloc(&dgpumodels, sizeof(model) * dmodels.size());
	cudaMemcpy(dgpumodels, dmodels.data(), sizeof(model) * dmodels.size(), cudaMemcpyHostToDevice);
	models_size = dmodels.size();
	world w{ dgpumodels, models_size };
	cudaMalloc(&dworld, sizeof(world));
	cudaMemcpy(dworld, &w, sizeof(world), cudaMemcpyHostToDevice);
	rgb* drgbs;
	cudaMalloc(&drgbs, sizeof(rgb) * width * height);
	ppixels = new pixels(width, height);
	ppixels->data = drgbs;
}

std::unique_ptr<RayTracer::rgb> RayTracer::render(input i, Projection proj_type) throw(RayTraceException)
{
	if (!dgpumodels || !ppixels) throw RayTraceException("init function not called");
	i.dworld = dworld;
	input* dinput = prepare_inputs(i);
	draw_frame(*ppixels, dinput, proj_type);
	cudaDeviceSynchronize();
	int size = (ppixels->width) * (ppixels->height);
	rgb* prgbs = new rgb[size];
	cudaMemcpy(prgbs, ppixels->data, sizeof(rgb) * size, cudaMemcpyDeviceToHost);
	cudaFree(dinput);
	return std::unique_ptr<rgb>(prgbs);
}

void RayTracer::clear()
{
	for (void* dshape : *p_all_shapes)
	{
		cudaFree(dshape);
	}
	for (unsigned char* dtexture : *p_all_textures)
	{
		cudaFree(dtexture);
	}
	cudaFree(dgpumodels);
	cudaFree(dworld);
	cudaFree(ppixels->data);
	delete p_all_shapes;
	delete ppixels;
}

void RayTracer::prepare_data(const std::shared_ptr<std::vector<Core::model*>> pmodels, std::vector<model>& dmodels)
{
	for (unsigned i = 0; i < pmodels->size(); i++)
	{
		Core::model* pmodel = (*pmodels)[i];
		RayTracer::model dmodel;
		if (pmodel->s_type == Core::shape_type::TRIANGLE)
		{
			std::vector<triangle> triangles;
			prepare_triangles(pmodel, &triangles);
			cudaMalloc(&dmodel.dshapes, sizeof(triangle) * pmodel->shapes_size);
			cudaMemcpy(dmodel.dshapes, triangles.data(), sizeof(triangle) * pmodel->shapes_size, cudaMemcpyHostToDevice);
		}
		else if (pmodel->s_type == Core::shape_type::SPHERE)
		{
			std::vector<sphere> spheres;
			prepare_spheres(pmodel, &spheres);
			cudaMalloc(&dmodel.dshapes, sizeof(sphere) * pmodel->shapes_size);
			cudaMemcpy(dmodel.dshapes, spheres.data(), sizeof(sphere) * pmodel->shapes_size, cudaMemcpyHostToDevice);
		}
		else if (pmodel->s_type == Core::shape_type::BOX)
		{
			cudaMalloc(&dmodel.dshapes, sizeof(Core::box) * pmodel->shapes_size);
			cudaMemcpy(dmodel.dshapes, pmodel->pshapes, sizeof(Core::box) * pmodel->shapes_size, cudaMemcpyHostToDevice);
		}
		dmodel.emissive_color = pmodel->emissive_color;
		dmodel.position = pmodel->position;
		dmodel.reflectivity = pmodel->reflectivity;
		dmodel.transparency = pmodel->transparency;
		dmodel.shapes_size = pmodel->shapes_size;
		dmodel.s_type = pmodel->s_type;
		dmodel.diffuse = get_texture(pmodel->diffuse);
		dmodel.specular = get_texture(pmodel->specular);
		dmodel.surface_color = pmodel->surface_color;
		dmodel.m_type = pmodel->m_type;
		dmodels.push_back(dmodel);
		p_all_shapes->push_back(dmodel.dshapes);
	}
}

RayTracer::texture RayTracer::get_texture(Core::texture core_texture)
{
	texture texture;
	cudaMalloc(&texture.dtextures, sizeof(unsigned char) * core_texture.width * core_texture.height);
	cudaMemcpy(texture.dtextures, core_texture.ptextures, sizeof(unsigned char) * core_texture.width * core_texture.height, cudaMemcpyHostToDevice);
	texture.width = core_texture.width;
	texture.height = core_texture.height;
	texture.channels = core_texture.channels;
	p_all_textures->push_back(texture.dtextures);
	return texture;
}

void RayTracer::prepare_triangles(const Core::model* pmodel, std::vector<RayTracer::triangle>* ptriangles)
{
	if (!ptriangles) return;
	for (unsigned i = 0; i < pmodel->shapes_size; i++)
	{
		Core::triangle* c_triangles = (Core::triangle*)pmodel->pshapes;
		Core::vertex* p_vertex_a = &c_triangles[i].a;
		p_vertex_a->position += pmodel->position;
		Core::vertex* p_vertex_b = &c_triangles[i].b;
		p_vertex_b->position += pmodel->position;
		Core::vertex* p_vertex_c = &c_triangles[i].c;
		p_vertex_c->position += pmodel->position;
		ptriangles->push_back(make_triangle(*p_vertex_a, *p_vertex_b, *p_vertex_c));
	}
}

void RayTracer::prepare_spheres(const Core::model* pmodel, std::vector<RayTracer::sphere>* pspheres)
{
	for (unsigned i = 0; i < pmodel->shapes_size; i++)
	{
		Core::sphere* c_spheres = (Core::sphere*)pmodel->pshapes;
		pspheres->push_back(sphere{ c_spheres[i].radius * c_spheres[i].radius, c_spheres[i].center });
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
	return triangle{ a.position,b.position,c.position,ab,bc,ca,a.texcoord,b.texcoord,c.texcoord,normal,plane_distance,area };
}

RayTracer::input* RayTracer::prepare_inputs(input& i)
{
	input* dinput;
	double* dtranslator;
	double* drotator;
	cudaMalloc(&dtranslator, sizeof(double) * i.translator.size);
	cudaMalloc(&drotator, sizeof(double) * i.rotator.size);
	cudaMemcpy(dtranslator, i.translator.pmatrix, sizeof(double) * i.translator.size, cudaMemcpyHostToDevice);
	cudaMemcpy(drotator, i.rotator.pmatrix, sizeof(double) * i.rotator.size, cudaMemcpyHostToDevice);
	i.translator.pmatrix = dtranslator;
	i.rotator.pmatrix = drotator;
	cudaMalloc(&dinput, sizeof(input));
	cudaMemcpy(dinput, &i, sizeof(input), cudaMemcpyHostToDevice);
	return dinput;
}