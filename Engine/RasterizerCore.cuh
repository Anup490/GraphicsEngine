#pragma once
#include "EngineCore.cuh"

namespace Engine
{
	struct model_data
	{
		model* dmodel;
		unsigned shape_count;
	};

	void draw_background(pixels pixels, Base::mat4 dirmatrix, Base::cubemap* dcubemap);
	void draw_frame(pixels pixels, const raster_input& dinput, model_data data);

	struct RasterizerCore
	{
		std::vector<unsigned char*>* p_all_textures = 0;
		std::vector<void*>* p_all_shapes = 0;
		std::vector<model_data>* p_all_models = 0;
		pixels* ppixels = 0;
		Base::cubemap* dcubemap = 0;
		double* pdirmatrix = 0;
		rgb* prgbs = 0;
		RasterizerCore(std::shared_ptr<std::vector<Base::model*>> pmodels, Base::cubemap* pcubemap, int width, int height);
		void prepare_data(const std::shared_ptr<std::vector<Base::model*>> pmodels);
		void prepare_triangles(const Base::model* pmodel, std::vector<triangle>* ptriangles);
		triangle make_triangle(Base::vertex a, Base::vertex b, Base::vertex c);
		Base::texture get_texture(Base::texture Base_texture);
		void prepare_cubemap(const Base::cubemap* pcubemap);
		raster_input prepare_input(const raster_input& i);
		Base::mat4 prepare_dirmatrix(const raster_input& i);
		~RasterizerCore();
	};
}