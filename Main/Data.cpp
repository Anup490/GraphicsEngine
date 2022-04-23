#include <string>
#include <memory>
#include <vector>
#include "FileReader.h"

Base::model* prepare_gltf_model_data(Base::model_info info) throw(FileReadException);
Base::texture get_texture(const char* file_path);
void delete_texture(Base::model* pmodel);
Base::model* prepare_sphere(const Base::sphere& sphere, const Base::vec3& surface_color, const double& smoothness, const double& metallicity, const double& transparency, const char* tex_path);
Base::model* prepare_box(const Base::box& box, const Base::vec3& surface_color, const double& smoothness, const double& metallicity, const double& transparency, const char* tex_path);


std::shared_ptr<std::vector<Base::model*>> prepare_data(Base::model*& pcamera)
{
	std::shared_ptr<std::vector<Base::model*>> pmodels(new std::vector<Base::model*>);
	//Base::model* pmodel = prepare_gltf_model_data({ "D:/Projects/C++/3DImporter/Assets/airplane/scene.gltf", Base::vec3{0.0,0.0,-3.0} });
	Base::model* psurface = prepare_box(Base::box{ Base::vec3{ -50.0, -40.0, -70.0 }, Base::vec3{ 100.0, -30.0, 20.0 } }, Base::vec3{ 0.13, 0.29, 0.45 }, 0.0, 0.0, 0.0, "");
	
	Base::model* psphere1 = prepare_sphere(Base::sphere{ 4.0, Base::vec3{ 5.0, -26.0, -10.0 } }, Base::vec3{ 1.0, 1.0, 1.0 }, 0.2, 0.9, 0.0, "D:/Projects/C++/3DImporter/Assets/jupiter/moon_baseColor.jpg");	
	Base::model* psphere2 = prepare_sphere(Base::sphere{ 4.0, Base::vec3{ 20.0, -26.0, -10.0 } }, Base::vec3{ 1.0, 1.0, 1.0 }, 0.9, 0.1, 0.7, "D:/Projects/C++/3DImporter/Assets/windows/diffuse.png");
	Base::model* psphere3 = prepare_sphere(Base::sphere{ 4.0, Base::vec3{ 35.0, -26.0, -10.0 } }, Base::vec3{ 1.0, 1.0, 1.0 }, 0.9, 0.8, 0.1, "D:/Projects/C++/opengl-tutorials-main/Resources/YoutubeOpenGL 30 - Bloom/textures/diffuse.png");

	Base::model* pbox1 = prepare_box(Base::box{ Base::vec3{ 0.0, -30.0, -30.0 }, Base::vec3{ 10.0, -20.0, -40.0 } }, Base::vec3{ 1.0, 1.0, 1.0 }, 0.3, 0.8, 0.0, "D:/Projects/C++/3DImporter/Assets/crow/diffuse.png");
	Base::model* pbox2 = prepare_box(Base::box{ Base::vec3{ 15.0, -30.0, -30.0 }, Base::vec3{ 25.0, -20.0, -40.0 } }, Base::vec3{ 1.0, 1.0, 1.0 }, 0.9, 0.1, 0.8, "D:/Projects/C++/OpenGLExercise/happy.png");
	Base::model* pbox3 = prepare_box(Base::box{ Base::vec3{ 30.0, -30.0, -30.0 }, Base::vec3{ 40.0, -20.0, -40.0 } }, Base::vec3{ 1.0, 1.0, 1.0 }, 0.0, 0.1, 0.0, "D:/Projects/C++/3DImporter/Assets/wall.png");

	Base::model* plight = new Base::model;
	plight->position = Base::vec3{ 75.0, 100.0, -100.0 };
	plight->emissive_color = Base::vec3{ 1.0, 1.0, 1.0 };
	plight->m_type = Base::model_type::LIGHT;

	pcamera = new Base::model;
	pcamera->m_type = Base::model_type::CAMERA;
	
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

Base::model* prepare_sphere(const Base::sphere& sphere, const Base::vec3& surface_color, const double& smoothness, const double& metallicity, const double& transparency, const char* tex_path)
{
	Base::model* pmodel = new Base::model;
	const unsigned sphere_count = 1;
	Base::sphere* pspheres = new Base::sphere[sphere_count];
	pspheres[0] =sphere;
	pmodel->position = pspheres[0].center;
	pmodel->emissive_color = Base::vec3{};
	pmodel->surface_color = surface_color;
	pmodel->pshapes = pspheres;
	pmodel->shapes_size = sphere_count;
	pmodel->s_type = Base::shape_type::SPHERE;
	pmodel->m_type = Base::model_type::OBJECT;
	pmodel->smoothness = smoothness;
	pmodel->metallicity = metallicity;
	pmodel->transparency = transparency;
	pmodel->diffuse = get_texture(tex_path);
	return pmodel;
}

Base::model* prepare_box(const Base::box& box, const Base::vec3& surface_color, const double& smoothness, const double& metallicity, const double& transparency, const char* tex_path)
{
	Base::model* pmodel = new Base::model;
	const unsigned sphere_count = 1;
	Base::box* pboxes = new Base::box[sphere_count];
	pboxes[0] = box;
	pmodel->position = pboxes[0].center;
	pmodel->emissive_color = Base::vec3{};
	pmodel->surface_color = surface_color;
	pmodel->pshapes = pboxes;
	pmodel->shapes_size = sphere_count;
	pmodel->s_type = Base::shape_type::BOX;
	pmodel->m_type = Base::model_type::OBJECT;
	pmodel->smoothness = smoothness;
	pmodel->metallicity = metallicity;
	pmodel->transparency = transparency;
	pmodel->diffuse = get_texture(tex_path);
	return pmodel;
}

void delete_data(std::shared_ptr<std::vector<Base::model*>> pmodels)
{
	for (Base::model* pmodel : *pmodels)
	{
		if (pmodel->m_type != Base::model_type::CAMERA)
		{
			delete_texture(pmodel);
			delete pmodel;
		}
	}
}