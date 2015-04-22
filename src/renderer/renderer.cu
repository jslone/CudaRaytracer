#include <cstring>
#include <iostream>
#include "renderer.h"

namespace acr
{
	struct DevParams
	{
		Scene *scene;
		char sceneData[sizeof(Scene)];
		uint32_t x,y;
	};

	__constant__
	DevParams devParams;

	Renderer::Renderer(const Renderer::Args &args)
		: title(args.title)
		, dim(args.dim)
	{
		if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0)
		{
			std::cerr << "SDL_InitSubSystem Error: " << SDL_GetError() << std::endl;
			exit(EXIT_FAILURE);
		}

		window = SDL_CreateWindow(title, args.pos.x, args.pos.y,
								  dim.x, dim.y, 0);
		if (window == nullptr)
		{
			std::cerr << "SDL_CreateWindow Error: " << SDL_GetError() << std::endl;
			exit(EXIT_FAILURE);
		}

		renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED
		                                        | SDL_RENDERER_PRESENTVSYNC);
		if (renderer == nullptr)
		{
			std::cerr << "SDL_CreateRenderer Error: " << SDL_GetError() << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	Renderer::~Renderer()
	{
		if (renderer)
		{
			SDL_DestroyRenderer(renderer);
		}
		if (window)
		{
			SDL_DestroyWindow(window);
		}
	}

	void Renderer::loadScene(const Scene &scene)
	{
		DevParams params;
		params.scene = (Scene*)(&devParams.sceneData[0]);
		std::memcpy(params.sceneData, &scene, sizeof(Scene));
		params.x = dim.x;
		params.y = dim.y;

		cudaMemcpyToSymbol(&devParams, &params, sizeof(DevParams));
	}

	__global__
	void scatterTrace()
	{
		int x = blockIdx.x * gridDim.x + threadIdx.x;
		int y = blockIdx.y * gridDim.y + threadIdx.y;
		int sample = blockIdx.z * gridDim.z + threadIdx.z;


	}

	void Renderer::render()
	{
		// call kernel to render pixels then draw to screen
	}

}
