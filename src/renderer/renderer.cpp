#include <iostream>
#include "renderer.h"

namespace acr {

  Renderer::Renderer() {
    if(SDL_Init(SDL_INIT_VIDEO) != 0) {
      std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
      exit(1);
    }
  }

  Renderer::~Renderer() {
  }

}
