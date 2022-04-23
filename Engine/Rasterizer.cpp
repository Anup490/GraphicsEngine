#include "pch.h"
#include <memory>
#include <vector>
#include "Rasterizer.h"
#include "RasterizerCore.cuh"
#include "Vector.cuh"

namespace Engine
{
	Rasterizer::Rasterizer(std::shared_ptr<std::vector<Base::model*>> pmodels, Base::cubemap* pcubemap, int width, int height)
	{
		if (((width % 32) != 0) || ((height % 32) != 0)) throw RasterizeException("width or height is not a multiple of 32");
		if (((width < 640) != 0) || ((height < 480) != 0)) throw RasterizeException("minimum acceptable resolution is 640x480");
		pcore = new RasterizerCore;
		pcore->p_all_models = new std::vector<model*>;
		pcore->p_all_shapes = new std::vector<void*>;
		pcore->p_all_textures = new std::vector<unsigned char*>();
		pcore->prepare_data(pmodels);
		rgb* drgbs;
		cudaMalloc(&drgbs, sizeof(rgb) * width * height);
		pcore->ppixels = new pixels(width, height);
		pcore->ppixels->data = drgbs;
	}

	std::unique_ptr<Engine::rgb> Rasterizer::render(raster_input i)
	{
		if (!pcore->ppixels) throw RasterizeException("init function not called");
		raster_input* dinput = pcore->prepare_inputs(i);
		draw_background(*pcore->ppixels, dinput, pcore->dcubemap);
		cudaDeviceSynchronize();
		for (model* dmodel : *pcore->p_all_models)
		{
			draw_frame(*pcore->ppixels, dinput, dmodel);
			cudaDeviceSynchronize();
		}
		int size = (pcore->ppixels->width) * (pcore->ppixels->height);
		rgb* prgbs = new rgb[size];
		cudaMemcpy(prgbs, pcore->ppixels->data, sizeof(rgb) * size, cudaMemcpyDeviceToHost);
		cudaFree(i.view.pmatrix);
		cudaFree(i.projection.pmatrix);
		cudaFree(dinput);
		return std::unique_ptr<rgb>(prgbs);
	}

	Rasterizer::~Rasterizer()
	{
		for (void* dshape : *pcore->p_all_shapes)
			cudaFree(dshape);
		for (unsigned char* dtexture : *pcore->p_all_textures)
			cudaFree(dtexture);
		for (model* dmodel : *pcore->p_all_models)
			cudaFree(dmodel);
		cudaFree(pcore->dcubemap);
		cudaFree(pcore->ppixels->data);
		delete pcore->p_all_shapes;
		delete pcore->p_all_textures;
		delete pcore->p_all_models;
		delete pcore->ppixels;
	}

	void RasterizerCore::prepare_data(const std::shared_ptr<std::vector<Base::model*>> pmodels)
	{
		for (unsigned i = 0; i < pmodels->size(); i++)
		{
			Base::model* pmodel = (*pmodels)[i];
			model model, * dmodel;
			if (pmodel->s_type == Base::shape_type::TRIANGLE)
			{
				std::vector<triangle> triangles;
				prepare_triangles(pmodel, &triangles);
				cudaMalloc(&model.dshapes, sizeof(triangle) * pmodel->shapes_size);
				cudaMemcpy(model.dshapes, triangles.data(), sizeof(triangle) * pmodel->shapes_size, cudaMemcpyHostToDevice);
			}
			model.emissive_color = pmodel->emissive_color;
			model.position = pmodel->position;
			model.smoothness = pmodel->smoothness;
			model.transparency = pmodel->transparency;
			model.metallicity = pmodel->metallicity;
			model.shapes_size = pmodel->shapes_size;
			model.s_type = pmodel->s_type;
			model.diffuse = get_texture(pmodel->diffuse);
			model.specular = get_texture(pmodel->specular);
			model.surface_color = pmodel->surface_color;
			model.m_type = pmodel->m_type;
			cudaMalloc(&dmodel, sizeof(model));
			cudaMemcpy(dmodel, &model, sizeof(model), cudaMemcpyHostToDevice);
			p_all_models->push_back(dmodel);
			p_all_shapes->push_back(model.dshapes);
			if (pmodel->m_type == Base::model_type::CAMERA) camera_index = i;
		}
	}

	void RasterizerCore::prepare_triangles(const Base::model* pmodel, std::vector<triangle>* ptriangles)
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

	triangle RasterizerCore::make_triangle(Base::vertex a, Base::vertex b, Base::vertex c)
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

	Base::texture RasterizerCore::get_texture(Base::texture Base_texture)
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

	void RasterizerCore::prepare_cubemap(Base::cubemap* pcubemap)
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
	}

	raster_input* RasterizerCore::prepare_inputs(raster_input& i)
	{
		raster_input* dinput;
		double* dtranslator;
		double* drotator;
		cudaMalloc(&dtranslator, sizeof(double) * i.view.size);
		cudaMalloc(&drotator, sizeof(double) * i.projection.size);
		cudaMemcpy(dtranslator, i.view.pmatrix, sizeof(double) * i.view.size, cudaMemcpyHostToDevice);
		cudaMemcpy(drotator, i.projection.pmatrix, sizeof(double) * i.projection.size, cudaMemcpyHostToDevice);
		i.view.pmatrix = dtranslator;
		i.projection.pmatrix = drotator;
		cudaMalloc(&dinput, sizeof(raster_input));
		cudaMemcpy(dinput, &i, sizeof(raster_input), cudaMemcpyHostToDevice);
		return dinput;
	}
}