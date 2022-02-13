#include <iostream>
#include <memory>
#include "FileReader.h"

void main()
{
	std::cout << "Preparing model data" << std::endl;
	try
	{
		std::shared_ptr<Core::model> pmodel = prepare_gltf_model_data("D:/Projects/C++/3DImporter/Assets/airplane/scene.gltf");
		std::cout << "Data extracted" << std::endl;
	}
	catch (FileReadException& e)
	{
		std::cout << e.what() << std::endl;
	}
}