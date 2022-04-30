#pragma once
#include "EngineCore.cuh"

namespace Engine
{
	void draw_frame(pixels pixels, const world* dworld, const raytrace_input& input);

	struct RayTracerCore
	{
		world* dworld = 0;
		model* dgpumodels = 0;
		Base::cubemap* dcubemap = 0;
		unsigned models_size = 0;
		pixels* ppixels = 0;
		std::vector<void*>* p_all_shapes = 0;
		std::vector<unsigned char*>* p_all_textures = 0;
		rgb* prgbs = 0;
		int camera_index = -1;
		RayTracerCore(std::shared_ptr<std::vector<Base::model*>> pmodels, Base::cubemap* pcubemap, int width, int height);
		void prepare_data(const std::shared_ptr<std::vector<Base::model*>> pmodels, std::vector<model>& dmodels);
		Base::texture get_texture(Base::texture Base_texture);
		void prepare_triangles(const Base::model* pmodel, std::vector<triangle>* ptriangles);
		void prepare_spheres(const Base::model* pmodel, std::vector<sphere>* pspheres);
		triangle make_triangle(Base::vertex a, Base::vertex b, Base::vertex c);
		raytrace_input prepare_input(const raytrace_input& i);
		Base::cubemap* prepare_cubemap(Base::cubemap* pcubemap);
		void update_camera(const raytrace_input& i);
		~RayTracerCore();
	};
}
