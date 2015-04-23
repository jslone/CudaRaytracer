#ifndef _SDL_MANAGER_H_
#define _SDL_MANAGER_H_

#include "SDL.h"

namespace acr
{
	class SDL
	{
	public:
		SDL() { SDL_Init(0); }
		~SDL() { SDL_Quit(); }
	};
}

#endif //_SDL_MANAGER_H_
