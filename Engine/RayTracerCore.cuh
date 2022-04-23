#pragma once
#include "EngineCore.cuh"

namespace Engine
{
	void draw_frame(pixels pixels, raytrace_input* dinput, Projection proj_type);

	struct RayTracerCore
	{
		world* dworld = 0;
		model* dgpumodels = 0;
		Base::cubemap* dcubemap = 0;
		unsigned models_size = 0;
		pixels* ppixels = 0;
		std::vector<void*>* p_all_shapes = 0;
		std::vector<unsigned char*>* p_all_textures = 0;
		int camera_index = -1;
		void prepare_data(const std::shared_ptr<std::vector<Base::model*>> pmodels, std::vector<model>& dmodels);
		Base::texture get_texture(Base::texture Base_texture);
		void prepare_triangles(const Base::model* pmodel, std::vector<triangle>* ptriangles);
		void prepare_spheres(const Base::model* pmodel, std::vector<sphere>* pspheres);
		triangle make_triangle(Base::vertex a, Base::vertex b, Base::vertex c);
		raytrace_input* prepare_inputs(raytrace_input& i);
		Base::cubemap* prepare_cubemap(Base::cubemap* pcubemap);
		void update_camera(raytrace_input& i);
	};
}
