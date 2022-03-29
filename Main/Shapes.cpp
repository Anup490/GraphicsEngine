#include <string>
#include <memory>
#include "FileReader.h"

std::unique_ptr<Core::model> prepare_spheres()
{
	Core::model* pmodel = new Core::model;
	const unsigned sphere_count = 1;
	Core::sphere* pspheres = new Core::sphere[sphere_count];
	pspheres[0] = Core::sphere{ 4.0, Core::vec3{ 0.0, 0.0, -20.0 }};
	pmodel->position = pspheres[0].center;
	pmodel->emissive_color = Core::vec3{};
	pmodel->surface_color = Core::vec3{ 1.00, 0.32, 0.36 };
	pmodel->pshapes = pspheres;
	pmodel->shapes_size = sphere_count;
	pmodel->s_type = Core::shape_type::SPHERE;
	pmodel->m_type = Core::model_type::OBJECT;


	/*
	pspheres[0] = Core::sphere(vec3(0.0f, -10004.0f, -20.0f), 10000.0f, vec3(0.20f, 0.20f, 0.20f), 0.0f, 0.0f, vec3(0.0f));
	pspheres[1] = Core::sphere(vec3(0.0f, 0.0f, -20.0f), 4.0f, vec3(1.00f, 0.32f, 0.36f), 1.0f, 0.5f, vec3(0.0f));
	pspheres[2] = Core::sphere(vec3(5.0f, -1.0f, -15.0f), 2.0f, vec3(0.90f, 0.76f, 0.46f), 1.0f, 0.0f, vec3(0.0f));
	pspheres[3] = Core::sphere(vec3(5.0f, 0.0f, -25.0f), 3.0f, vec3(0.65f, 0.77f, 0.97f), 1.0f, 0.0f, vec3(0.0f));
	pspheres[4] = Core::sphere(vec3(-5.5f, 0.0f, -15.0f), 3.0f, vec3(0.90f, 0.90f, 0.90f), 1.0f, 0.0f, vec3(0.0f));
	pspheres[5] = Core::sphere(vec3(0.0f, 20.0f, -30.0f), 3.0f, vec3(1.0f, 1.0f, 1.0f), 0.0f, 0.0f, vec3(1.0f));
	*/

	return std::unique_ptr<Core::model>(pmodel);
}

std::unique_ptr<Core::model> prepare_boxes()
{
	Core::model* pmodel = new Core::model;
	const unsigned sphere_count = 1;
	Core::box* pboxes = new Core::box[sphere_count];
	pboxes[0] = Core::box{ Core::vec3{ 0.0, 0.0, -20.0 }, Core::vec3{ 10.0, 10.0, -30.0 } };
	pmodel->position = pboxes[0].center;
	pmodel->emissive_color = Core::vec3{};
	pmodel->surface_color = Core::vec3{ 1.00, 0.32, 0.36 };
	pmodel->pshapes = pboxes;
	pmodel->shapes_size = sphere_count;
	pmodel->s_type = Core::shape_type::BOX;
	pmodel->m_type = Core::model_type::OBJECT;
	return std::unique_ptr<Core::model>(pmodel);
}