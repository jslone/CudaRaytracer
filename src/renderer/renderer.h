#ifndef _RENDERER_H_
#define _RENDERER_H_

#include "SDL.h"
#include "math/math.h"
#include "scene/scene.h"

namespace acr {

  class Renderer {
    public:
      
      struct Args {
        const char *title;
        math::vec2 pos,dim;
      };

      Renderer(const Args &args);
      ~Renderer();

      void loadScene(const Scene &scene);
      void render();
    private:
      SDL_Window *window;
      SDL_Renderer *renderer;
  };

} // namespace acr

#endif //_RENDERER_H_
