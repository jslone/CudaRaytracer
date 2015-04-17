#ifndef _SCENE_H_
#define _SCENE_H_

#include "assimp/scene.h"
#include "geometry/geometry.h"

namespace acr {
  
  class Object {
  };

  class Scene {
    public:
      struct Args {
        const char* filePath;
      };

      Scene(const Args &args);
      ~Scene();      
    private:
  };

} // namespace acr

#endif //_SCENE_H_
