#ifndef _SCENE_H_
#define _SCENE_H_

#include "assimp/scene.h"
#include "geometry/geometry.h"
#include "math/math.h"
#include <unordered_map>
#include <string>

namespace acr {
  
  class Object : Shape {
    public:
      virtual bool intersect(const Ray &r, HitInfo &info);

    	int meshIndex;
    	int numChildren;
      int index;
      int parentIndex;
    	Object* parent;
      Object** children;
    	math::mat4 globalTransform;
    	math::mat4 localTransform;
      math::mat4 globalNormalTransform;

    	math::mat4 globalInverseTransform;
      math::mat4 globalInverseNormalTransform;
  };

  class Camera {
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

  class Scene : Shape {
    public:
      struct Args {
        const char* filePath;
      };

      Scene(const Args &args);
      ~Scene();

      virtual bool intersect(const Ray& r, HitInfo &info);
    private:
      std::unordered_map<std::string, Light*> light_map;
      std::unordered_map<std::string, Camera*> camera_map;
      void loadScene(const aiScene* scene);
      Object* loadObject(aiNode* node, Object* parent);
      Light** loadLights(const aiScene* scene);
      Material* loadMaterials(const aiScene* scene);
      Camera loadCamera(aiCamera* cam);
      Mesh* loadMeshes(const aiScene* scene);
      void getMathMatrix(aiMatrix4x4& aiMatrix, math::mat4& mathMat);
      Object* rootObject;
      Object* objects;
      Camera camera;
      Light** lights;
      Material* materials;
      Mesh* meshes;
      int numLights;
      int numMaterials;
      int numMeshes;
      int numObjects;
  };

} // namespace acr

#endif //_SCENE_H_
