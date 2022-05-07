#pragma once
#include "EngineCore.cuh"

namespace Engine
{
	struct model_data
	{
		model *dmodel, *dcamera, *dlights;
		unsigned lights_count, shape_count;
		triangle *dtview, *dtndc, *dtraster;
	};

	void draw_background(pixels pixels, Base::mat4 dirmatrix, Base::cubemap* dcubemap);
	void draw_frame(pixels pixels, const raster_input& input, model_data* ddata, unsigned shape_count);

	struct RasterizerCore
	{
		const double triangle_min_area = 0.00005;
		std::vector<unsigned char*>* p_all_textures = 0;
		std::vector<void*>* p_all_shapes = 0;
		std::vector<model_data>* p_all_models = 0;
		pixels* ppixels = 0;
		Base::cubemap* dcubemap = 0;
		double* pdirmatrix = 0;
		rgb* prgbs = 0;
		model* dlights = 0;
		unsigned lights_count = 0;
		model* dcamera = 0;
		RasterizerCore(std::shared_ptr<std::vector<Base::model*>> pmodels, Base::cubemap* pcubemap, int width, int height);
		void prepare_data(const std::shared_ptr<std::vector<Base::model*>> pmodels);
		void prepare_lights(const std::shared_ptr<std::vector<Base::model*>> pmodels);
		void prepare_camera(const std::shared_ptr<std::vector<Base::model*>> pmodels);
		void prepare_triangles(const Base::model* pmodel, std::vector<triangle>* ptriangles);
		void split_and_store_triangle(triangle& t, std::vector<triangle>* ptriangles);
		triangle make_triangle(Base::vertex a, Base::vertex b, Base::vertex c);
		Base::texture get_texture(Base::texture Base_texture);
		void prepare_cubemap(const Base::cubemap* pcubemap);
		raster_input prepare_input(const raster_input& i);
		Base::mat4 prepare_dirmatrix(const raster_input& i);
		void update_camera(const Base::model* pcamera);
		~RasterizerCore();
	};
}
