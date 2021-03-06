#include "json.h"
#include "Base.h"
#include "FileReader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include "stb_image.h"

std::vector<unsigned char>* get_data(nlohmann::json& JSON, const char* file_path);
void traverse_node(nlohmann::json& JSON, unsigned nextNode, std::vector<unsigned char>* pdata, Base::model* pmodel, Base::model_info& info);
void load_mesh(nlohmann::json& JSON, unsigned int indMesh, std::vector<unsigned char>* pdata, Base::model* pmodel, Base::model_info& info);
std::vector<float>* get_floats(nlohmann::json& accessor, nlohmann::json& JSON, std::vector<unsigned char>* pdata);
std::vector<Base::vec3>* group_floats_for_vec3(std::vector<float>* pfloatvec);
std::vector<Base::vec3>* group_floats_for_vec2(std::vector<float>* pfloatvec);
std::vector<Base::vertex>* get_vertices(std::vector<Base::vec3>* ppositions, std::vector<Base::vec3>* pnormals, std::vector<Base::vec3>* ptextUVs);
std::vector<unsigned>* get_indices(nlohmann::json& accessor, nlohmann::json& JSON, std::vector<unsigned char>* pdata);
void set_diffuse_texture(const char* file, const nlohmann::json& JSON, Base::texture& texture_data);
void set_specular_texture(const char* file, const nlohmann::json& JSON, Base::texture& texture_data);
void triangulate(Base::model* pmodel, std::vector<Base::vertex>* pvertices, std::vector<unsigned>* pindices, Base::vec3& position);

Base::model* prepare_gltf_model_data(Base::model_info info)
{
	std::string json_string = extract_file(info.file_path);
	if (json_string.empty()) throw FileReadException("Error reading gltf file");
	nlohmann::json json_data = nlohmann::json::parse(json_string);
	std::vector<unsigned char>* pdata = get_data(json_data, info.file_path);
	Base::model* pmodel = new Base::model;
	traverse_node(json_data, 0, pdata, pmodel, info);
	delete pdata;
	if(!pmodel->diffuse.ptextures) throw FileReadException(std::string("Error loading textures at : ").append(info.file_path));
	return pmodel;
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

void delete_texture(Base::model* pmodel)
{
	if(pmodel->diffuse.ptextures) stbi_image_free(pmodel->diffuse.ptextures);
	if(pmodel->specular.ptextures) stbi_image_free(pmodel->specular.ptextures);
}

Base::texture get_texture(const char* file_path)
{
	Base::texture texture;
	stbi_set_flip_vertically_on_load(true);
	texture.ptextures = stbi_load(file_path, &(texture.width), &(texture.height), &(texture.channels), 0);
	return texture;
}

std::unique_ptr<Base::cubemap> prepare_cubemap(const char* file_path)
{
	Base::cubemap* pcubemap = new Base::cubemap;
	std::string path(file_path);
	if (path.substr(path.length() - 1) != "/") path.append("/");
	std::string faces[] = { "left.jpg", "right.jpg", "bottom.jpg", "top.jpg", "front.jpg", "back.jpg" };
	pcubemap->left = get_texture((path + faces[0]).c_str());
	pcubemap->right = get_texture((path + faces[1]).c_str());
	pcubemap->bottom = get_texture((path + faces[2]).c_str());
	pcubemap->top = get_texture((path + faces[3]).c_str());
	pcubemap->front = get_texture((path + faces[4]).c_str());
	pcubemap->back = get_texture((path + faces[5]).c_str());
	return std::unique_ptr<Base::cubemap>(pcubemap);
}

void delete_cubemap(std::unique_ptr<Base::cubemap>& pcubemap)
{
	if (pcubemap->left.ptextures) stbi_image_free(pcubemap->left.ptextures);
	if (pcubemap->right.ptextures) stbi_image_free(pcubemap->right.ptextures);
	if (pcubemap->top.ptextures) stbi_image_free(pcubemap->top.ptextures);
	if (pcubemap->bottom.ptextures) stbi_image_free(pcubemap->bottom.ptextures);
	if (pcubemap->front.ptextures) stbi_image_free(pcubemap->front.ptextures);
	if (pcubemap->back.ptextures) stbi_image_free(pcubemap->back.ptextures);
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

void traverse_node(nlohmann::json& JSON, unsigned nextNode, std::vector<unsigned char>* pdata, Base::model* pmodel, Base::model_info& info)
{
	nlohmann::json node = JSON["nodes"][nextNode];
	bool mesh_not_found = true;
	if (node.find("mesh") != node.end())
	{
		mesh_not_found = false;
		load_mesh(JSON, node["mesh"], pdata, pmodel, info);
	}
	if (mesh_not_found)
	{
		traverse_node(JSON, ++nextNode, pdata, pmodel, info);
	}
}

void load_mesh(nlohmann::json& JSON, unsigned int indMesh, std::vector<unsigned char>* pdata, Base::model* pmodel, Base::model_info& info)
{
	unsigned int pos_acc_ind = JSON["meshes"][indMesh]["primitives"][0]["attributes"]["POSITION"];
	unsigned int normal_acc_ind = JSON["meshes"][indMesh]["primitives"][0]["attributes"]["NORMAL"];
	unsigned int tex_acc_ind = JSON["meshes"][indMesh]["primitives"][0]["attributes"]["TEXCOORD_0"];
	unsigned int ind_acc_ind = JSON["meshes"][indMesh]["primitives"][0]["indices"];

	std::vector<float>* pposvec = get_floats(JSON["accessors"][pos_acc_ind], JSON, pdata);
	std::vector<Base::vec3>* ppositions = group_floats_for_vec3(pposvec);
	std::vector<float>* pnormalvec = get_floats(JSON["accessors"][normal_acc_ind], JSON, pdata);
	std::vector<Base::vec3>* pnormals = group_floats_for_vec3(pnormalvec);
	std::vector<float>* ptexvec = get_floats(JSON["accessors"][tex_acc_ind], JSON, pdata);
	std::vector<Base::vec3>* ptexUVs = group_floats_for_vec2(ptexvec);

	std::vector<Base::vertex>* pvertices = get_vertices(ppositions, pnormals, ptexUVs);
	std::vector<unsigned>* pindices = get_indices(JSON["accessors"][ind_acc_ind], JSON, pdata);
	triangulate(pmodel, pvertices, pindices, info.position);
	set_diffuse_texture(info.file_path, JSON, pmodel->diffuse);
	set_specular_texture(info.file_path, JSON, pmodel->specular);

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

std::vector<Base::vec3>* group_floats_for_vec3(std::vector<float>* pfloatvec)
{
	std::vector<Base::vec3>* pvectors = new std::vector<Base::vec3>();
	for (unsigned i = 0; i < pfloatvec->size();)
	{
		float x = pfloatvec->at(i++);
		float y = pfloatvec->at(i++);
		float z = pfloatvec->at(i++);
		pvectors->push_back(Base::vec3{ x, y, z });
	}
	return pvectors;
}

std::vector<Base::vec3>* group_floats_for_vec2(std::vector<float>* pfloatvec)
{
	std::vector<Base::vec3>* pvectors = new std::vector<Base::vec3>();
	for (unsigned i = 0; i < pfloatvec->size();)
	{
		float x = pfloatvec->at(i++);
		float y = pfloatvec->at(i++);
		pvectors->push_back(Base::vec3{ x, y, 0.0 });
	}
	return pvectors;
}

std::vector<Base::vertex>* get_vertices(std::vector<Base::vec3>* ppositions, std::vector<Base::vec3>* pnormals, std::vector<Base::vec3>* ptextUVs)
{
	std::vector<Base::vertex>* pvertices = new std::vector<Base::vertex>();;
	for (unsigned i = 0; i < ppositions->size(); i++)
	{
		pvertices->push_back(Base::vertex{ ppositions->at(i), pnormals->at(i), ptextUVs->at(i) });
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

void set_diffuse_texture(const char* file, const nlohmann::json& JSON, Base::texture& texture_data)
{
	std::vector<float>* ptextures = new std::vector<float>();
	std::string fileStr = std::string(file);
	std::string fileDirectory = fileStr.substr(0, fileStr.find_last_of('/') + 1);
	for (unsigned int i = 0; i < JSON["images"].size(); i++)
	{
		std::string tex_name= JSON["images"][i]["uri"];
		std::string tex_path = fileDirectory + tex_name;
		if (tex_name.find("baseColor") != std::string::npos || tex_name.find("diffuse") != std::string::npos)
		{
			stbi_set_flip_vertically_on_load(true);
			texture_data.ptextures = stbi_load(tex_path.c_str(), &texture_data.width, &texture_data.height, &texture_data.channels, 0);
		}
	}
}

void set_specular_texture(const char* file, const nlohmann::json& JSON, Base::texture& texture_data)
{
	std::vector<float>* ptextures = new std::vector<float>();
	std::string fileStr = std::string(file);
	std::string fileDirectory = fileStr.substr(0, fileStr.find_last_of('/') + 1);
	for (unsigned int i = 0; i < JSON["images"].size(); i++)
	{
		std::string tex_name = JSON["images"][i]["uri"];
		std::string tex_path = fileDirectory + tex_name;
		if (tex_name.find("metallicRoughness") != std::string::npos || tex_name.find("specular") != std::string::npos)
		{
			stbi_set_flip_vertically_on_load(true);
			texture_data.ptextures = stbi_load(tex_path.c_str(), &texture_data.width, &texture_data.height, &texture_data.channels, 0);
		}
	}
}

void triangulate(Base::model* pmodel, std::vector<Base::vertex>* pvertices, std::vector<unsigned>* pindices, Base::vec3& position)
{
	if (!pvertices || !pindices) return;
	pmodel->shapes_size = pindices->size() / 3;
	pmodel->pshapes = new Base::triangle[pmodel->shapes_size];
	pmodel->s_type = Base::shape_type::TRIANGLE;
	pmodel->m_type = Base::model_type::OBJECT;
	pmodel->surface_color = Base::vec3{ 1.0, 1.0, 1.0 };
	pmodel->position = position;
	unsigned t = 0;
	for (unsigned i = 0; i < pindices->size();)
	{
		Base::vertex a = (*pvertices)[(*pindices)[i++]];
		Base::vertex b = (*pvertices)[(*pindices)[i++]];
		Base::vertex c = (*pvertices)[(*pindices)[i++]];
		Base::triangle* ptriangles = (Base::triangle*)(pmodel->pshapes);
		ptriangles[t++] = Base::triangle{ a, b, c };
	}
}