#ifndef _SCENE_H_
#define _SCENE_H_

#include "assimp/scene.h"
#include "geometry/geometry.h"
#include "math/math.h"

namespace acr {
  
  class Object {
    public:
    	int meshIndex;
    	int numChildren;
    	Object* parent;
      Object** children;
    	math::mat4 globalTransform;
    	math::mat4 localTransform;
    	math::mat4 globalInverseTransform;
  };

  class Camera{
    public:
      float aspectRatio;
      float horizontalFOV;
      math::mat4 globalTransform;
  };

  class Light{
    public:
      float attConstant;
      float attLinear;
      float attQuadratic;
      Color3 ambient;
      Color3 diffuse;
      Color3 specular;
      math::vec3 position;

      virtual float getFlux(math::vec3 position) = 0;
  };

  class DirectionalLight : public Light{
    public:
      math::vec3 direction;

      virtual float getFlux(math::vec3 position){
        return 0;
      }
  };

  class PointLight : public Light{
    public:
      virtual float getFlux(math::vec3 position){
        return 0;
      }
  };

  class SpotLight : public Light{
    public:
      float innerConeAngle;
      float outerConeAngle;
      math::vec3 direction;

      virtual float getFlux(math::vec3 position){
        return 0;
      }
  };

  class Scene {
    public:
      struct Args {
        const char* filePath;
      };

      Scene(const Args &args);
      ~Scene();      
    private:
      void loadScene(const aiScene* scene);
      Object* loadObject(aiNode* node, Object* parent);
      Light** loadLights(const aiScene* scene);
      Material* loadMaterials(const aiScene* scene);
      Camera loadCamera(aiCamera* cam);
      void getMathMatrix(aiMatrix4x4& aiMatrix, math::mat4& mathMat);
      Object* rootObject;
      Camera camera;
      Light** lights;
      Material* materials;
      int numLights;
      int numMaterials;
  };

} // namespace acr

#endif //_SCENE_H_
