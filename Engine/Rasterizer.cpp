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
		pcore = new RasterizerCore(pmodels, pcubemap, width, height);
	}

	Engine::rgb* Rasterizer::render(const raster_input& i, const Base::model* pcamera)
	{
		if (!pcore->ppixels) throw RasterizeException("init function not called");
		Base::mat4 dirmatrix = pcore->prepare_dirmatrix(i);
		raster_input input = pcore->prepare_input(i);
		pcore->update_camera(pcamera);
		draw_background(*pcore->ppixels, dirmatrix, pcore->dcubemap);
		cudaDeviceSynchronize();
		for (model_data data : *pcore->p_all_models)
		{
			draw_frame(*pcore->ppixels, input, data, pcore->dcamera, pcore->dlights, pcore->lights_count);
			cudaDeviceSynchronize();
		}
		int size = (pcore->ppixels->width) * (pcore->ppixels->height);
		cudaMemcpy(pcore->prgbs, pcore->ppixels->data, sizeof(rgb) * size, cudaMemcpyDeviceToHost);
		cudaFree(input.view.pmatrix);
		cudaFree(input.projection.pmatrix);
		return pcore->prgbs;
	}

	Rasterizer::~Rasterizer()
	{
		delete pcore;
	}

	RasterizerCore::RasterizerCore(std::shared_ptr<std::vector<Base::model*>> pmodels, Base::cubemap* pcubemap, int width, int height)
	{
		p_all_models = new std::vector<model_data>;
		p_all_shapes = new std::vector<void*>;
		p_all_textures = new std::vector<unsigned char*>();
		prepare_data(pmodels);
		prepare_lights(pmodels);
		prepare_camera(pmodels);
		prepare_cubemap(pcubemap);
		cudaMalloc(&pdirmatrix, sizeof(double) * 16);
		rgb* drgbs;
		cudaMalloc(&drgbs, sizeof(rgb) * width * height);
		ppixels = new pixels(width, height);
		ppixels->data = drgbs;
		int size = width * height;
		prgbs = new rgb[size];
	}

	void RasterizerCore::prepare_data(const std::shared_ptr<std::vector<Base::model*>> pmodels)
	{
		for (unsigned i = 0; i < pmodels->size(); i++)
		{
			Base::model* pmodel = (*pmodels)[i];
			if (pmodel->m_type != Base::model_type::OBJECT) continue;
			model model, * dmodel;
			std::vector<triangle> triangles;
			if (pmodel->s_type == Base::shape_type::TRIANGLE)
			{
				prepare_triangles(pmodel, &triangles);
				cudaMalloc(&model.dshapes, sizeof(triangle) * triangles.size());
				cudaMemcpy(model.dshapes, triangles.data(), sizeof(triangle) * triangles.size(), cudaMemcpyHostToDevice);
			}
			model.emissive_color = pmodel->emissive_color;
			model.position = pmodel->position;
			model.smoothness = pmodel->smoothness;
			model.transparency = pmodel->transparency;
			model.metallicity = pmodel->metallicity;
			model.shapes_size = triangles.size();
			model.s_type = pmodel->s_type;
			model.diffuse = get_texture(pmodel->diffuse);
			model.specular = get_texture(pmodel->specular);
			model.surface_color = pmodel->surface_color;
			model.m_type = pmodel->m_type;
			cudaMalloc(&dmodel, sizeof(model));
			cudaMemcpy(dmodel, &model, sizeof(model), cudaMemcpyHostToDevice);
			p_all_models->push_back(model_data{ dmodel, unsigned(triangles.size()) });
			p_all_shapes->push_back(model.dshapes);
		}
	}

	void RasterizerCore::prepare_lights(const std::shared_ptr<std::vector<Base::model*>> pmodels)
	{
		if (pmodels->size() == 0) return;
		std::vector<model> lights;
		for (unsigned i = 0; i < pmodels->size(); i++)
		{
			Base::model* pmodel = (*pmodels)[i];
			if (pmodel->m_type != Base::model_type::LIGHT) continue;
			model model;
			model.emissive_color = pmodel->emissive_color;
			model.position = pmodel->position;
			model.surface_color = pmodel->surface_color;
			model.m_type = pmodel->m_type;
			lights.push_back(model);
		}
		cudaMalloc(&dlights, sizeof(model) * lights.size());
		cudaMemcpy(dlights, lights.data(), sizeof(model) * lights.size(), cudaMemcpyHostToDevice);
		lights_count = lights.size();
	}


	void RasterizerCore::prepare_camera(const std::shared_ptr<std::vector<Base::model*>> pmodels)
	{
		for (Base::model* pmodel : *pmodels)
		{
			if (pmodel->m_type == Base::model_type::CAMERA)
			{
				model camera;
				camera.position = pmodel->position;
				camera.m_type = pmodel->m_type;
				cudaMalloc(&dcamera, sizeof(model));
				cudaMemcpy(dcamera, &camera, sizeof(model), cudaMemcpyHostToDevice);
				break;
			}
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
			triangle t = make_triangle(*p_vertex_a, *p_vertex_b, *p_vertex_c);
			split_and_store_triangle(t, ptriangles);
		}
	}

	void RasterizerCore::split_and_store_triangle(triangle& t, std::vector<triangle>* ptriangles)
	{
		if (t.area > triangle_min_area)
		{
			double ab_length = length(t.ab);
			double bc_length = length(t.bc);
			double ca_length = length(t.ca);
			double max_length = maximum(ab_length, bc_length, ca_length);
			Base::vertex a_vertex{ t.a, t.normal, t.a_tex };
			Base::vertex b_vertex{ t.b, t.normal, t.b_tex };
			Base::vertex c_vertex{ t.c, t.normal, t.c_tex };
			if (equal(ab_length, max_length))
			{
				Base::vec3 m{ (t.a.x + t.b.x) / 2.0, (t.a.y + t.b.y) / 2.0, (t.a.z + t.b.z) / 2.0, };
				Base::vec3 m_tex{ (t.a_tex.x + t.b_tex.x) / 2.0, (t.a_tex.y + t.b_tex.y) / 2.0, (t.a_tex.z + t.b_tex.z) / 2.0, };
				Base::vertex m_vertex{ m, t.normal, m_tex };
				triangle t1 = make_triangle(a_vertex, m_vertex, c_vertex);
				split_and_store_triangle(t1, ptriangles);
				triangle t2 = make_triangle(m_vertex, b_vertex, c_vertex);
				split_and_store_triangle(t2, ptriangles);
			}
			else if (equal(bc_length, max_length))
			{
				Base::vec3 m{ (t.b.x + t.c.x) / 2.0, (t.b.y + t.c.y) / 2.0, (t.b.z + t.c.z) / 2.0, };
				Base::vec3 m_tex{ (t.b_tex.x + t.c_tex.x) / 2.0, (t.b_tex.y + t.c_tex.y) / 2.0, (t.b_tex.z + t.c_tex.z) / 2.0, };
				Base::vertex m_vertex{ m, t.normal, m_tex };
				triangle t1 = make_triangle(a_vertex, b_vertex, m_vertex);
				split_and_store_triangle(t1, ptriangles);
				triangle t2 = make_triangle(a_vertex, m_vertex, c_vertex);
				split_and_store_triangle(t2, ptriangles);
			}
			else
			{
				Base::vec3 m{ (t.c.x + t.a.x) / 2.0, (t.c.y + t.a.y) / 2.0, (t.c.z + t.a.z) / 2.0, };
				Base::vec3 m_tex{ (t.c_tex.x + t.a_tex.x) / 2.0, (t.c_tex.y + t.a_tex.y) / 2.0, (t.c_tex.z + t.a_tex.z) / 2.0, };
				Base::vertex m_vertex{ m, t.normal, m_tex };
				triangle t1 = make_triangle(a_vertex, b_vertex, m_vertex);
				split_and_store_triangle(t1, ptriangles);
				triangle t2 = make_triangle(m_vertex, b_vertex, c_vertex);
				split_and_store_triangle(t2, ptriangles);
			}
		}
		else
		{
			ptriangles->push_back(t);
		}
	}

	triangle RasterizerCore::make_triangle(Base::vertex a, Base::vertex b, Base::vertex c)
	{
		Base::vec3 ab = b.position - a.position;
		Base::vec3 bc = c.position - b.position;
		Base::vec3 ca = a.position - c.position;
		Base::vec3 normal = cross(ab, bc);
		double area = length(normal) / 2.0;
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

	void RasterizerCore::prepare_cubemap(const Base::cubemap* pcubemap)
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

	raster_input RasterizerCore::prepare_input(const raster_input& i)
	{
		raster_input input;
		cudaMalloc(&input.view.pmatrix, sizeof(double) * i.view.size);
		cudaMalloc(&input.projection.pmatrix, sizeof(double) * i.projection.size);
		cudaMemcpy(input.view.pmatrix, i.view.pmatrix, sizeof(double) * i.view.size, cudaMemcpyHostToDevice);
		cudaMemcpy(input.projection.pmatrix, i.projection.pmatrix, sizeof(double) * i.projection.size, cudaMemcpyHostToDevice);
		return input;
	}

	Base::mat4 RasterizerCore::prepare_dirmatrix(const raster_input& i)
	{
		Base::mat4 dirmatrix;
		dirmatrix.pmatrix = new double[16];
		for (unsigned r = 0; r < 4; r++)
		{
			for (unsigned c = 0; c < 4; c++)
			{
				unsigned m = r * 4 + c;
				dirmatrix.pmatrix[c*4 + r] = ((m + 1) % 4 == 0) ? 0 : i.view.pmatrix[m];
			}
		}
		dirmatrix.pmatrix[15] = 1;
		cudaMemcpy(pdirmatrix, dirmatrix.pmatrix, sizeof(double) * 16, cudaMemcpyHostToDevice);
		delete[] dirmatrix.pmatrix;
		dirmatrix.pmatrix = pdirmatrix;
		return dirmatrix;
	}

	void RasterizerCore::update_camera(const Base::model* pcamera)
	{
		if (dcamera)
		{
			model camera;
			camera.position = pcamera->position;
			camera.m_type = pcamera->m_type;
			cudaMemcpy(dcamera, &camera, sizeof(model), cudaMemcpyHostToDevice);
		}
	}

	RasterizerCore::~RasterizerCore()
	{
		for (void* dshape : *p_all_shapes)
			cudaFree(dshape);
		for (unsigned char* dtexture : *p_all_textures)
			cudaFree(dtexture);
		for (model_data data : *p_all_models)
			cudaFree(data.dmodel);
		cudaFree(dcubemap);
		cudaFree(pdirmatrix);
		cudaFree(ppixels->data);
		cudaFree(dlights);
		cudaFree(dcamera);
		delete p_all_shapes;
		delete p_all_textures;
		delete p_all_models;
		delete ppixels;
		delete prgbs;
	}
}