#include "json.h"
#include "Core.h"
#include "FileReader.h"
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>

std::vector<unsigned char>* get_data(nlohmann::json& JSON, const char* file_path);
std::string extract_file(const char* path);
void traverse_node(nlohmann::json& JSON, unsigned nextNode, std::vector<unsigned char>* pdata, Core::model* pmodel);
void load_mesh(nlohmann::json& JSON, unsigned int indMesh, std::vector<unsigned char>* pdata, Core::model* pmodel);
std::vector<float>* get_floats(nlohmann::json& accessor, nlohmann::json& JSON, std::vector<unsigned char>* pdata);
std::vector<Core::vec3>* group_floats_for_vec3(std::vector<float>* pfloatvec);
std::vector<Core::vec3>* group_floats_for_vec2(std::vector<float>* pfloatvec);
std::vector<Core::vertex>* get_vertices(std::vector<Core::vec3>* ppositions, std::vector<Core::vec3>* pnormals, std::vector<Core::vec3>* ptextUVs);
std::vector<unsigned>* get_indices(nlohmann::json& accessor, nlohmann::json& JSON, std::vector<unsigned char>* pdata);

std::unique_ptr<Core::model> prepare_gltf_model_data(const char* file_path)
{
	std::string json_string = extract_file(file_path);
	if (json_string.empty())
	{
		throw FileReadException("Error reading gltf file");
	}
	nlohmann::json json_data = nlohmann::json::parse(json_string);
	std::vector<unsigned char>* pdata = get_data(json_data, file_path);
	Core::model* pmodel = new Core::model;
	traverse_node(json_data, 0, pdata, pmodel);
	delete pdata;
	return std::unique_ptr<Core::model>(pmodel);
}

std::vector<unsigned char>* get_data(nlohmann::json& JSON, const char* file_path)
{
	std::string bytesText;
	std::string uri = JSON["buffers"][0]["uri"];
	std::string fileStr = std::string(file_path);
	std::string fileDirectory = fileStr.substr(0, fileStr.find_last_of('/') + 1);
	bytesText = extract_file((fileDirectory + uri).c_str());
	std::vector<unsigned char>* pdata = new std::vector<unsigned char>(bytesText.begin(), bytesText.end());
	return pdata;
}

std::string extract_file(const char* path)
{
	std::string contents = "";
	std::ifstream file_input_stream;
	std::stringstream string_stream;
	try
	{
		file_input_stream.open(path, std::ios::binary);
		string_stream << file_input_stream.rdbuf();
		file_input_stream.close();
		contents = string_stream.str();
		return contents;
	}
	catch (const std::ifstream::failure& f)
	{
		std::cout << "Failure reading :: " << path << std::endl;
		std::cout << "Error :: " << f.what() << std::endl;
	}
	return contents;
}

void traverse_node(nlohmann::json& JSON, unsigned nextNode, std::vector<unsigned char>* pdata, Core::model* pmodel)
{
	nlohmann::json node = JSON["nodes"][nextNode];
	bool mesh_not_found = true;
	if (node.find("mesh") != node.end())
	{
		mesh_not_found = false;
		load_mesh(JSON, node["mesh"], pdata, pmodel);
	}
	if (mesh_not_found)
	{
		traverse_node(JSON, ++nextNode, pdata, pmodel);
	}
}

void load_mesh(nlohmann::json& JSON, unsigned int indMesh, std::vector<unsigned char>* pdata, Core::model* pmodel)
{
	unsigned int pos_acc_ind = JSON["meshes"][indMesh]["primitives"][0]["attributes"]["POSITION"];
	unsigned int normal_acc_ind = JSON["meshes"][indMesh]["primitives"][0]["attributes"]["NORMAL"];
	unsigned int tex_acc_ind = JSON["meshes"][indMesh]["primitives"][0]["attributes"]["TEXCOORD_0"];
	unsigned int ind_acc_ind = JSON["meshes"][indMesh]["primitives"][0]["indices"];

	std::vector<float>* pposvec = get_floats(JSON["accessors"][pos_acc_ind], JSON, pdata);
	std::vector<Core::vec3>* ppositions = group_floats_for_vec3(pposvec);
	std::vector<float>* pnormalvec = get_floats(JSON["accessors"][normal_acc_ind], JSON, pdata);
	std::vector<Core::vec3>* pnormals = group_floats_for_vec3(pnormalvec);
	std::vector<float>* ptexvec = get_floats(JSON["accessors"][tex_acc_ind], JSON, pdata);
	std::vector<Core::vec3>* ptexUVs = group_floats_for_vec2(ptexvec);

	pmodel->pvertices = get_vertices(ppositions, pnormals, ptexUVs);
	pmodel->pindices = get_indices(JSON["accessors"][ind_acc_ind], JSON, pdata);

	delete pposvec;
	delete ppositions;
	delete pnormalvec;
	delete pnormals;
	delete ptexvec;
	delete ptexUVs;
}

std::vector<float>* get_floats(nlohmann::json& accessor, nlohmann::json& JSON, std::vector<unsigned char>* pdata)
{
	std::vector<float>* pfloatvec = new std::vector<float>();

	unsigned int buffViewInd = accessor.value("bufferView", 1);
	unsigned int count = accessor["count"];
	unsigned int accByteOffset = accessor.value("byteOffset", 0);
	std::string type = accessor["type"];

	nlohmann::json bufferView = JSON["bufferViews"][buffViewInd];
	unsigned int byteOffset = bufferView["byteOffset"];

	unsigned int numPerVert;
	if (type == "SCALAR") numPerVert = 1;
	else if (type == "VEC2") numPerVert = 2;
	else if (type == "VEC3") numPerVert = 3;
	else if (type == "VEC4") numPerVert = 4;
	else throw std::invalid_argument("Type is invalid (not SCALAR, VEC2, VEC3 or VEC4)");

	unsigned int beginningOfData = byteOffset + accByteOffset;
	unsigned int lengthOfData = count * 4 * numPerVert;
	for (unsigned int i = beginningOfData; i < beginningOfData + lengthOfData; i)
	{
		unsigned char bytes[] = { (*pdata)[i++], (*pdata)[i++], (*pdata)[i++], (*pdata)[i++] };
		float value;
		std::memcpy(&value, bytes, sizeof(float));
		pfloatvec->push_back(value);
	}
	return pfloatvec;
}

std::vector<Core::vec3>* group_floats_for_vec3(std::vector<float>* pfloatvec)
{
	std::vector<Core::vec3>* pvectors = new std::vector<Core::vec3>();
	for (unsigned i = 0; i < pfloatvec->size();)
	{
		float x = pfloatvec->at(i++);
		float y = pfloatvec->at(i++);
		float z = pfloatvec->at(i++);
		pvectors->push_back(Core::vec3{ x, y, z - 3.0 });
	}
	return pvectors;
}

std::vector<Core::vec3>* group_floats_for_vec2(std::vector<float>* pfloatvec)
{
	std::vector<Core::vec3>* pvectors = new std::vector<Core::vec3>();
	for (unsigned i = 0; i < pfloatvec->size();)
	{
		float x = pfloatvec->at(i++);
		float y = pfloatvec->at(i++);
		pvectors->push_back(Core::vec3{ x, y, 0.0 });
	}
	return pvectors;
}

std::vector<Core::vertex>* get_vertices(std::vector<Core::vec3>* ppositions, std::vector<Core::vec3>* pnormals, std::vector<Core::vec3>* ptextUVs)
{
	std::vector<Core::vertex>* pvertices = new std::vector<Core::vertex>();;
	for (unsigned i = 0; i < ppositions->size(); i++)
	{
		pvertices->push_back(Core::vertex{ ppositions->at(i), pnormals->at(i), ptextUVs->at(i) });
	}
	return pvertices;
}

std::vector<unsigned>* get_indices(nlohmann::json& accessor, nlohmann::json& JSON, std::vector<unsigned char>* pdata)
{
	std::vector<unsigned>* pindices = new std::vector<unsigned>();

	unsigned int buffViewInd = accessor.value("bufferView", 0);
	unsigned int count = accessor["count"];
	unsigned int accByteOffset = accessor.value("byteOffset", 0);
	unsigned int componentType = accessor["componentType"];

	nlohmann::json bufferView = JSON["bufferViews"][buffViewInd];
	unsigned int byteOffset = bufferView["byteOffset"];

	unsigned int beginningOfData = byteOffset + accByteOffset;
	if (componentType == 5125)
	{
		for (unsigned int i = beginningOfData; i < byteOffset + accByteOffset + count * 4; i)
		{
			unsigned char bytes[] = { (*pdata)[i++], (*pdata)[i++], (*pdata)[i++], (*pdata)[i++] };
			unsigned int value;
			std::memcpy(&value, bytes, sizeof(unsigned int));
			pindices->push_back((unsigned)value);
		}
	}
	else if (componentType == 5123)
	{
		for (unsigned int i = beginningOfData; i < byteOffset + accByteOffset + count * 2; i)
		{
			unsigned char bytes[] = { (*pdata)[i++], (*pdata)[i++] };
			unsigned short value;
			std::memcpy(&value, bytes, sizeof(unsigned short));
			pindices->push_back((unsigned)value);
		}
	}
	else if (componentType == 5122)
	{
		for (unsigned int i = beginningOfData; i < byteOffset + accByteOffset + count * 2; i)
		{
			unsigned char bytes[] = { (*pdata)[i++], (*pdata)[i++] };
			short value;
			std::memcpy(&value, bytes, sizeof(short));
			pindices->push_back((unsigned)value);
		}
	}
	return pindices;
}