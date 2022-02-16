#include <iostream>
#include <memory>
#include <vector>
#include "FileReader.h"
#include "RayTracer.h"

#include <fstream>
void write_to_file(Core::vec3* pixels, int width, int height);

void main()
{
	std::cout << "Preparing model data" << std::endl;
	bool init_called = false;
	try
	{
		std::unique_ptr<Core::model> pmodel = prepare_gltf_model_data("D:/Projects/C++/3DImporter/Assets/airplane/scene.gltf");
		Core::model light_model{ Core::vec3{}, Core::vec3{1.0, 1.0, 1.0} };
		std::cout << "Data extracted" << std::endl;
		std::cout << "Rendering scene" << std::endl;
		std::shared_ptr<std::vector<Core::model*>> pmodels(new std::vector<Core::model*>);
		pmodels->push_back(pmodel.get());
		pmodels->push_back(&light_model);
		RayTracer::init(pmodels, 640, 480);
		init_called = true;
		std::unique_ptr<Core::vec3> ppixels = RayTracer::render(90.0, RayTracer::Projection::PERSPECTIVE);
		write_to_file(ppixels.get(), 640, 480);
		std::cout << "Scene rendered" << std::endl;
	}
	catch (std::exception& e)
	{
		std::cout << "Exception thrown :: "<< e.what() << std::endl;
	}
	if(init_called) RayTracer::clear();
}

void write_to_file(Core::vec3* pixels, int width, int height)
{
	std::ofstream ofs("gengine.ppm", std::ios::out | std::ios::binary);
	ofs << "P6\n" << width << " " << height << "\n255\n";
	for (unsigned i = 0; i < width * height; ++i) {
		ofs << (unsigned char)(std::min(double(1), pixels[i].x) * 255) <<
			(unsigned char)(std::min(double(1), pixels[i].y) * 255) <<
			(unsigned char)(std::min(double(1), pixels[i].z) * 255);
	}
	ofs.close();
}