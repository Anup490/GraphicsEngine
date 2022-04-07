#include <string>
#include <memory>
#include <vector>
#include "FileReader.h"

std::unique_ptr<Core::model> prepare_gltf_model_data(Core::model_info info) throw(FileReadException);
Core::texture get_texture(const char* file_path);
void delete_texture(Core::model* pmodel);

std::shared_ptr<std::vector<Core::model*>> prepare_data(Core::model*& pcamera)
{
	std::shared_ptr<std::vector<Core::model*>> pmodels(new std::vector<Core::model*>);
	//std::unique_ptr<Core::model> pmodel = prepare_gltf_model_data({ "D:/Projects/C++/3DImporter/Assets/airplane/scene.gltf", Core::vec3{0.0,0.0,-3.0} });

	Core::model* psphere = new Core::model;
	const unsigned sphere_count = 1;
	Core::sphere* pspheres = new Core::sphere[sphere_count];
	pspheres[0] = Core::sphere{ 4.0, Core::vec3{ 0.0, 10.0, -20.0 } };
	psphere->position = pspheres[0].center;
	psphere->emissive_color = Core::vec3{};
	psphere->surface_color = Core::vec3{ 1.0, 1.0, 1.0 };
	psphere->pshapes = pspheres;
	psphere->shapes_size = sphere_count;
	psphere->s_type = Core::shape_type::SPHERE;
	psphere->m_type = Core::model_type::OBJECT;
	psphere->smoothness = 0.2;
	psphere->metallicity = 0.9;
	psphere->diffuse = get_texture("D:/Projects/C++/3DImporter/Assets/jupiter/moon_baseColor.jpg");

	Core::model* pbox = new Core::model;
	const unsigned box_count = 1;
	Core::box* pboxes = new Core::box[box_count];
	pboxes[0] = Core::box{ Core::vec3{ 0.0, 0.0, -40.0 }, Core::vec3{ 10.0, 10.0, -30.0 } };
	pbox->position = pboxes[0].center;
	pbox->emissive_color = Core::vec3{};
	pbox->surface_color = Core::vec3{ 1.0, 1.0, 1.0 };
	pbox->pshapes = pboxes;
	pbox->shapes_size = box_count;
	pbox->s_type = Core::shape_type::BOX;
	pbox->m_type = Core::model_type::OBJECT;
	pbox->smoothness = 0.3;
	pbox->metallicity = 0.3;
	pbox->diffuse = get_texture("D:/Projects/C++/3DImporter/Assets/crow/diffuse.png");

	Core::model* plight = new Core::model;
	Core::sphere* p_light_spheres = new Core::sphere[1];
	p_light_spheres[0] = Core::sphere{ 2.0, Core::vec3{ 75.0, 100.0, -100.0  } };
	plight->position = p_light_spheres[0].center;
	plight->surface_color = Core::vec3{ 1.0, 1.0, 0.0 };
	plight->emissive_color = Core::vec3{ 1.0, 1.0, 1.0 };
	plight->pshapes = p_light_spheres;
	plight->shapes_size = 1;
	plight->m_type = Core::model_type::LIGHT;

	pcamera = new Core::model;
	pcamera->m_type = Core::model_type::CAMERA;
	//pmodels->push_back(pmodel.get());
	pmodels->push_back(psphere);
	pmodels->push_back(pbox);
	pmodels->push_back(plight);
	pmodels->push_back(pcamera);

	return pmodels;
}

void delete_data(std::shared_ptr<std::vector<Core::model*>> pmodels)
{
	for (Core::model* pmodel : *pmodels)
	{
		if (pmodel->m_type != Core::model_type::CAMERA)
		{
			delete_texture(pmodel);
			delete pmodel;
		}
	}
}