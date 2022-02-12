#include <iostream>
#include "FileReader.h"

void main()
{
	std::cout << "Preparing model data" << std::endl;
	prepare_model_data("D:/Projects/C++/3DImporter/Assets/airplane/scene.gltf");
	std::cout << "Data written to files" << std::endl;
}