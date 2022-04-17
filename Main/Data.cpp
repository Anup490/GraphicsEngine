#include <string>
#include <memory>
#include <vector>
#include "FileReader.h"

Core::model* prepare_gltf_model_data(Core::model_info info) throw(FileReadException);
Core::texture get_texture(const char* file_path);
void delete_texture(Core::model* pmodel);
Core::model* prepare_sphere(const Core::sphere& sphere, const Core::vec3& surface_color, const double& smoothness, const double& metallicity, const double& transparency, const char* tex_path);
Core::model* prepare_box(const Core::box& box, const Core::vec3& surface_color, const double& smoothness, const double& metallicity, const double& transparency, const char* tex_path);


std::shared_ptr<std::vector<Core::model*>> prepare_data(Core::model*& pcamera)
{
	std::shared_ptr<std::vector<Core::model*>> pmodels(new std::vector<Core::model*>);
	//Core::model* pmodel = prepare_gltf_model_data({ "D:/Projects/C++/3DImporter/Assets/airplane/scene.gltf", Core::vec3{0.0,0.0,-3.0} });
	Core::model* psurface = prepare_box(Core::box{ Core::vec3{ -50.0, -40.0, -70.0 }, Core::vec3{ 100.0, -30.0, 20.0 } }, Core::vec3{ 0.13, 0.29, 0.45 }, 0.0, 0.0, 0.0, "");
	
	Core::model* psphere1 = prepare_sphere(Core::sphere{ 4.0, Core::vec3{ 5.0, -26.0, -10.0 } }, Core::vec3{ 1.0, 1.0, 1.0 }, 0.2, 0.9, 0.0, "D:/Projects/C++/3DImporter/Assets/jupiter/moon_baseColor.jpg");	
	Core::model* psphere2 = prepare_sphere(Core::sphere{ 4.0, Core::vec3{ 20.0, -26.0, -10.0 } }, Core::vec3{ 1.0, 1.0, 1.0 }, 0.9, 0.1, 0.7, "D:/Projects/C++/3DImporter/Assets/windows/diffuse.png");
	Core::model* psphere3 = prepare_sphere(Core::sphere{ 4.0, Core::vec3{ 35.0, -26.0, -10.0 } }, Core::vec3{ 1.0, 1.0, 1.0 }, 0.9, 0.8, 0.1, "D:/Projects/C++/opengl-tutorials-main/Resources/YoutubeOpenGL 30 - Bloom/textures/diffuse.png");

	Core::model* pbox1 = prepare_box(Core::box{ Core::vec3{ 0.0, -30.0, -30.0 }, Core::vec3{ 10.0, -20.0, -40.0 } }, Core::vec3{ 1.0, 1.0, 1.0 }, 0.3, 0.8, 0.0, "D:/Projects/C++/3DImporter/Assets/crow/diffuse.png");
	Core::model* pbox2 = prepare_box(Core::box{ Core::vec3{ 15.0, -30.0, -30.0 }, Core::vec3{ 25.0, -20.0, -40.0 } }, Core::vec3{ 1.0, 1.0, 1.0 }, 0.9, 0.1, 0.8, "D:/Projects/C++/OpenGLExercise/happy.png");
	Core::model* pbox3 = prepare_box(Core::box{ Core::vec3{ 30.0, -30.0, -30.0 }, Core::vec3{ 40.0, -20.0, -40.0 } }, Core::vec3{ 1.0, 1.0, 1.0 }, 0.0, 0.1, 0.0, "D:/Projects/C++/3DImporter/Assets/wall.png");

	Core::model* plight = new Core::model;
	plight->position = Core::vec3{ 75.0, 100.0, -100.0 };
	plight->emissive_color = Core::vec3{ 1.0, 1.0, 1.0 };
	plight->m_type = Core::model_type::LIGHT;

	pcamera = new Core::model;
	pcamera->m_type = Core::model_type::CAMERA;
	
	pmodels->push_back(pcamera);
	pmodels->push_back(plight);	
	//pmodels->push_back(pmodel);
	pmodels->push_back(psurface);
	pmodels->push_back(psphere1);
	pmodels->push_back(psphere2);
	pmodels->push_back(psphere3);
	pmodels->push_back(pbox1);
	pmodels->push_back(pbox2);
	pmodels->push_back(pbox3);
	
	return pmodels;
}

Core::model* prepare_sphere(const Core::sphere& sphere, const Core::vec3& surface_color, const double& smoothness, const double& metallicity, const double& transparency, const char* tex_path)
{
	Core::model* pmodel = new Core::model;
	const unsigned sphere_count = 1;
	Core::sphere* pspheres = new Core::sphere[sphere_count];
	pspheres[0] =sphere;
	pmodel->position = pspheres[0].center;
	pmodel->emissive_color = Core::vec3{};
	pmodel->surface_color = surface_color;
	pmodel->pshapes = pspheres;
	pmodel->shapes_size = sphere_count;
	pmodel->s_type = Core::shape_type::SPHERE;
	pmodel->m_type = Core::model_type::OBJECT;
	pmodel->smoothness = smoothness;
	pmodel->metallicity = metallicity;
	pmodel->transparency = transparency;
	pmodel->diffuse = get_texture(tex_path);
	return pmodel;
}

Core::model* prepare_box(const Core::box& box, const Core::vec3& surface_color, const double& smoothness, const double& metallicity, const double& transparency, const char* tex_path)
{
	Core::model* pmodel = new Core::model;
	const unsigned sphere_count = 1;
	Core::box* pboxes = new Core::box[sphere_count];
	pboxes[0] = box;
	pmodel->position = pboxes[0].center;
	pmodel->emissive_color = Core::vec3{};
	pmodel->surface_color = surface_color;
	pmodel->pshapes = pboxes;
	pmodel->shapes_size = sphere_count;
	pmodel->s_type = Core::shape_type::BOX;
	pmodel->m_type = Core::model_type::OBJECT;
	pmodel->smoothness = smoothness;
	pmodel->metallicity = metallicity;
	pmodel->transparency = transparency;
	pmodel->diffuse = get_texture(tex_path);
	return pmodel;
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