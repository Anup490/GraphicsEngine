#include "pch.h"
#include "RayTracer.h"
#include "RayTracerCore.cuh"
#include "Vector.cuh"
#include "Matrix.cuh"
#include <memory>
#include <vector>

namespace Engine
{
	RayTracer::RayTracer(std::shared_ptr<std::vector<Base::model*>> pmodels, Base::cubemap* pcubemap, int width, int height)
	{
		if (((width % 32) != 0) || ((height % 32) != 0)) throw RayTraceException("width or height is not a multiple of 32");
		if (((width < 640) != 0) || ((height < 480) != 0)) throw RayTraceException("minimum acceptable resolution is 640x480");
		pcore = new RayTracerCore(pmodels, pcubemap, width, height);
	}

	rgb* RayTracer::render(const raytrace_input& i, Projection proj_type) throw(RayTraceException)
	{
		if (!pcore->dgpumodels || !pcore->ppixels) throw RayTraceException("init function not called");
		pcore->update_camera(i);
		raytrace_input input = pcore->prepare_input(i);
		draw_frame(*pcore->ppixels, pcore->dworld, input);
		cudaDeviceSynchronize();
		int size = (pcore->ppixels->width) * (pcore->ppixels->height);		
		cudaMemcpy(pcore->prgbs, pcore->ppixels->data, sizeof(rgb) * size, cudaMemcpyDeviceToHost);
		cudaFree(input.translator.pmatrix);
		cudaFree(input.rotator.pmatrix);
		return pcore->prgbs;
	}

	RayTracer::~RayTracer()
	{
		delete pcore;
	}

	RayTracerCore::RayTracerCore(std::shared_ptr<std::vector<Base::model*>> pmodels, Base::cubemap* pcubemap, int width, int height)
	{
		std::vector<model> dmodels;
		p_all_shapes = new std::vector<void*>;
		p_all_textures = new std::vector<unsigned char*>();
		prepare_data(pmodels, dmodels);
		cudaMalloc(&dgpumodels, sizeof(model) * dmodels.size());
		cudaMemcpy(dgpumodels, dmodels.data(), sizeof(model) * dmodels.size(), cudaMemcpyHostToDevice);
		models_size = dmodels.size();
		world w{ dgpumodels, models_size };
		w.dcubemap = prepare_cubemap(pcubemap);
		cudaMalloc(&dworld, sizeof(world));
		cudaMemcpy(dworld, &w, sizeof(world), cudaMemcpyHostToDevice);
		rgb* drgbs;
		cudaMalloc(&drgbs, sizeof(rgb) * width * height);
		ppixels = new pixels(width, height);
		ppixels->data = drgbs;
		int size = width * height;
		prgbs = new rgb[size];
	}

	void RayTracerCore::prepare_data(const std::shared_ptr<std::vector<Base::model*>> pmodels, std::vector<model>& dmodels)
	{
		for (unsigned i = 0; i < pmodels->size(); i++)
		{
			Base::model* pmodel = (*pmodels)[i];
			Engine::model dmodel;
			if (pmodel->s_type == Base::shape_type::TRIANGLE)
			{
				std::vector<triangle> triangles;
				prepare_triangles(pmodel, &triangles);
				cudaMalloc(&dmodel.dshapes, sizeof(triangle) * pmodel->shapes_size);
				cudaMemcpy(dmodel.dshapes, triangles.data(), sizeof(triangle) * pmodel->shapes_size, cudaMemcpyHostToDevice);
			}
			else if (pmodel->s_type == Base::shape_type::SPHERE)
			{
				std::vector<sphere> spheres;
				prepare_spheres(pmodel, &spheres);
				cudaMalloc(&dmodel.dshapes, sizeof(sphere) * pmodel->shapes_size);
				cudaMemcpy(dmodel.dshapes, spheres.data(), sizeof(sphere) * pmodel->shapes_size, cudaMemcpyHostToDevice);
			}
			else if (pmodel->s_type == Base::shape_type::BOX)
			{
				cudaMalloc(&dmodel.dshapes, sizeof(Base::box) * pmodel->shapes_size);
				cudaMemcpy(dmodel.dshapes, pmodel->pshapes, sizeof(Base::box) * pmodel->shapes_size, cudaMemcpyHostToDevice);
			}
			dmodel.emissive_color = pmodel->emissive_color;
			dmodel.position = pmodel->position;
			dmodel.smoothness = pmodel->smoothness;
			dmodel.transparency = pmodel->transparency;
			dmodel.metallicity = pmodel->metallicity;
			dmodel.shapes_size = pmodel->shapes_size;
			dmodel.s_type = pmodel->s_type;
			dmodel.diffuse = get_texture(pmodel->diffuse);
			dmodel.specular = get_texture(pmodel->specular);
			dmodel.surface_color = pmodel->surface_color;
			dmodel.m_type = pmodel->m_type;
			dmodels.push_back(dmodel);
			if (pmodel->m_type == Base::model_type::CAMERA) camera_index = i;
			p_all_shapes->push_back(dmodel.dshapes);
		}
	}

	Base::texture RayTracerCore::get_texture(Base::texture Base_texture)
	{
		Base::texture texture;
		cudaMalloc(&texture.ptextures, sizeof(unsigned char) * Base_texture.width * Base_texture.height * Base_texture.channels);
		cudaMemcpy(texture.ptextures, Base_texture.ptextures, sizeof(unsigned char) * Base_texture.width * Base_texture.height * Base_texture.channels, cudaMemcpyHostToDevice);
		texture.width = Base_texture.width;
		texture.height = Base_texture.height;
		texture.channels = Base_texture.channels;
		p_all_textures->push_back(texture.ptextures);
		return texture;
	}

	void RayTracerCore::prepare_triangles(const Base::model* pmodel, std::vector<triangle>* ptriangles)
	{
		if (!ptriangles) return;
		for (unsigned i = 0; i < pmodel->shapes_size; i++)
		{
			Base::triangle* c_triangles = (Base::triangle*)pmodel->pshapes;
			Base::vertex* p_vertex_a = &c_triangles[i].a;
			p_vertex_a->position += pmodel->position;
			Base::vertex* p_vertex_b = &c_triangles[i].b;
			p_vertex_b->position += pmodel->position;
			Base::vertex* p_vertex_c = &c_triangles[i].c;
			p_vertex_c->position += pmodel->position;
			ptriangles->push_back(make_triangle(*p_vertex_a, *p_vertex_b, *p_vertex_c));
		}
	}

	void RayTracerCore::prepare_spheres(const Base::model* pmodel, std::vector<sphere>* pspheres)
	{
		for (unsigned i = 0; i < pmodel->shapes_size; i++)
		{
			Base::sphere* c_spheres = (Base::sphere*)pmodel->pshapes;
			pspheres->push_back(sphere{ c_spheres[i].radius * c_spheres[i].radius, c_spheres[i].center });
		}
	}

	triangle RayTracerCore::make_triangle(Base::vertex a, Base::vertex b, Base::vertex c)
	{
		Base::vec3 ab = b.position - a.position;
		Base::vec3 bc = c.position - b.position;
		Base::vec3 ca = a.position - c.position;
		Base::vec3 normal = cross(ab, bc);
		double area = length(normal) / 2.0;
		normalize(normal);
		double plane_distance = dot(-normal, a.position);
		return triangle{ a.position,b.position,c.position,ab,bc,ca,a.texcoord,b.texcoord,c.texcoord,normal,plane_distance,area };
	}

	raytrace_input RayTracerCore::prepare_input(const raytrace_input& i)
	{
		raytrace_input input;
		input.fov = i.fov;
		input.near = i.near;
		input.far = i.far;
		input.proj_type = i.proj_type;
		double* dtranslator;
		double* drotator;
		cudaMalloc(&dtranslator, sizeof(double) * i.translator.size);
		cudaMalloc(&drotator, sizeof(double) * i.rotator.size);
		cudaMemcpy(dtranslator, i.translator.pmatrix, sizeof(double) * i.translator.size, cudaMemcpyHostToDevice);
		cudaMemcpy(drotator, i.rotator.pmatrix, sizeof(double) * i.rotator.size, cudaMemcpyHostToDevice);
		input.translator.pmatrix = dtranslator;
		input.rotator.pmatrix = drotator;
		return input;
	}

	Base::cubemap* RayTracerCore::prepare_cubemap(Base::cubemap* pcubemap)
	{
		Base::cubemap cubemap;
		cubemap.left = get_texture(pcubemap->left);
		cubemap.right = get_texture(pcubemap->right);
		cubemap.bottom = get_texture(pcubemap->bottom);
		cubemap.top = get_texture(pcubemap->top);
		cubemap.front = get_texture(pcubemap->front);
		cubemap.back = get_texture(pcubemap->back);
		cudaMalloc(&dcubemap, sizeof(Base::cubemap));
		cudaMemcpy(dcubemap, &cubemap, sizeof(Base::cubemap), cudaMemcpyHostToDevice);
		return dcubemap;
	}

	void RayTracerCore::update_camera(const raytrace_input& i)
	{
		if (camera_index >= 0)
		{
			model* pcamera = new model;
			model* dcamera = dgpumodels + camera_index;
			cudaMemcpy(pcamera, dcamera, sizeof(model), cudaMemcpyDeviceToHost);
			if (pcamera)
			{
				pcamera->position = i.translator * pcamera->position;
				cudaMemcpy(dcamera, pcamera, sizeof(model), cudaMemcpyHostToDevice);
			}
			delete pcamera;
		}
	}

	RayTracerCore::~RayTracerCore()
	{
		for (void* dshape : *p_all_shapes)
			cudaFree(dshape);
		for (unsigned char* dtexture : *p_all_textures)
			cudaFree(dtexture);
		cudaFree(dgpumodels);
		cudaFree(dcubemap);
		cudaFree(dworld);
		cudaFree(ppixels->data);
		delete p_all_shapes;
		delete ppixels;
		delete prgbs;
	}
}

