#include <string>
#include <memory>
#include "FileReader.h"

std::unique_ptr<Core::model> prepare_spheres()
{
	Core::model* pmodel = new Core::model;
	const unsigned sphere_count = 1;
	Core::sphere* pspheres = new Core::sphere[sphere_count];
	pspheres[0] = Core::sphere{ 4.0, Core::vec3{ 0.0, 10.0, -20.0 }};
	pmodel->position = pspheres[0].center;
	pmodel->emissive_color = Core::vec3{};
	pmodel->surface_color = Core::vec3{ 1.0, 1.0, 1.0 };
	pmodel->pshapes = pspheres;
	pmodel->shapes_size = sphere_count;
	pmodel->s_type = Core::shape_type::SPHERE;
	pmodel->m_type = Core::model_type::OBJECT;
	pmodel->reflectivity = 1.0;
	pmodel->metallicity = 1.0;
	pmodel->diffuse = get_texture("D:/Projects/C++/3DImporter/Assets/jupiter/moon_baseColor.jpg");
    return std::unique_ptr<Core::model>(pmodel);
}

std::unique_ptr<Core::model> prepare_boxes()
{
	Core::model* pmodel = new Core::model;
	const unsigned sphere_count = 1;
	Core::box* pboxes = new Core::box[sphere_count];
	pboxes[0] = Core::box{ Core::vec3{ 0.0, 0.0, -40.0 }, Core::vec3{ 10.0, 10.0, -30.0 } };
	pmodel->position = pboxes[0].center;
	pmodel->emissive_color = Core::vec3{};
	pmodel->surface_color = Core::vec3{ 1.0, 1.0, 1.0 };
	pmodel->pshapes = pboxes;
	pmodel->shapes_size = sphere_count;
	pmodel->s_type = Core::shape_type::BOX;
	pmodel->m_type = Core::model_type::OBJECT;
	pmodel->reflectivity = 1.0;
	pmodel->metallicity = 1.0;
	pmodel->diffuse = get_texture("D:/Projects/C++/3DImporter/Assets/crow/diffuse.png");
	return std::unique_ptr<Core::model>(pmodel);
}