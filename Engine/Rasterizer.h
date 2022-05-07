#pragma once
#include "Engine.h"
#include <exception>

namespace Engine
{
	struct RasterizeException : std::exception {
		const char* message;
		RasterizeException(const char* message) : message(message) {}
		char const* what() const override { return message; }
	};

	class Rasterizer
	{
		struct RasterizerCore* pcore;
	public:
		Rasterizer(std::shared_ptr<std::vector<Base::model*>> pmodels, Base::cubemap* pcubemap, int width, int height) throw(RasterizeException);
		pixels render(const raster_input& i, const Base::model* pcamera) throw(RasterizeException);
		~Rasterizer();
	};
}