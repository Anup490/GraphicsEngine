#pragma once
#include "EngineCore.cuh"

namespace Engine
{
	void draw_background(pixels pixels, raster_input* dinput, Base::cubemap* dcubemap);
	void draw_frame(pixels pixels, raster_input* dinput, model* dmodel);

	struct RasterizerCore
	{
		int camera_index = -1;
		std::vector<unsigned char*>* p_all_textures = 0;
		std::vector<void*>* p_all_shapes = 0;
		std::vector<model*>* p_all_models = 0;
		pixels* ppixels = 0;
		Base::cubemap* dcubemap = 0;
		void prepare_data(const std::shared_ptr<std::vector<Base::model*>> pmodels);
		void prepare_triangles(const Base::model* pmodel, std::vector<triangle>* ptriangles);
		triangle make_triangle(Base::vertex a, Base::vertex b, Base::vertex c);
		Base::texture get_texture(Base::texture Base_texture);
		void prepare_cubemap(Base::cubemap* pcubemap);
		raster_input* prepare_inputs(raster_input& i);
	};
}
