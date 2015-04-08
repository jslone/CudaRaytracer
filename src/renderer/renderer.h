#include "SDL.h"
#include "math/math.h"

#ifndef _RENDERER_H_
#define _RENDERER_H_

namespace acr {

  class Renderer {
    public:
      
      struct Args {
        const char *title;
        math::vec2 pos,dim;
      };

      Renderer(const Args &args);
      ~Renderer();
    private:
      SDL_Window *window;
      SDL_Renderer *renderer;
  };

} // namespace acr

#endif //_RENDERER_H_
