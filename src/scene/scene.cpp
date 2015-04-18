#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "scene.h"

namespace acr{
	Scene::Scene(const Scene::Args &args){
		Assimp::Importer importer;

		const aiScene* scene = importer.ReadFile(args.filePath, 
		        aiProcess_Triangulate            |
		        aiProcess_JoinIdenticalVertices  |
		        aiProcess_SortByPType);
		  
		// If the import failed, report it
		//if(!scene)
			//To error log: importer.GetErrorString();

		// Use the scene
		//DoTheSceneProcessing( scene);
		loadScene(scene);
	}

	Scene::~Scene(){
		//Todo
	}

	inline Color3 getColor3(aiColor3D aicol){
		return Color3(aicol.r, aicol.g, aicol.b);
	}

	inline math::vec3 getVec3(aiVector3D aivec){
		return math::vec3(aivec.x, aivec.y, aivec.z);
	}

	inline glm::tvec3<float> getTvec3(aiVector3D aivec){
		return glm::tvec3<float>(aivec.x, aivec.y, aivec.z);
	}

	void Scene::loadScene(const aiScene* scene){
		//scene->mMaterials[]	scene->mNumMaterials
		//scene->mMeshes[]		scene->mNumMeshes
		//scene->mTextures[]	scene->mNumTextures

		//Load camera
		camera = loadCamera(scene->mCameras[0]);

		//Load lights
		lights = loadLights(scene);

		//Load materials
		materials = loadMaterials(scene);

		//Load textures

		//Load meshes

		//Load object hierarchy
		rootObject = loadObject(scene->mRootNode, NULL);
	}

	//FIX THIS
	void Scene::getMathMatrix(aiMatrix4x4& aiMatrix, math::mat4& mathMat){
		for(int i = 0; i < 4; i++){
			for(int j = 0; j < 4; j++){
				mathMat[i][j] = aiMatrix[j][i];
			}
		}
	}

	Material* Scene::loadMaterials(const aiScene* scene){
		numMaterials = scene->mNumMaterials;
		Material* mats = new Material[numMaterials];

		for(int i = 0; i < numMaterials; i++){
			Material &mat = mats[i];
			aiMaterial* m = scene->mMaterials[i];

			aiColor3D diffuse, ambient, specular;
			m->Get(AI_MATKEY_COLOR_DIFFUSE,diffuse);
			m->Get(AI_MATKEY_COLOR_AMBIENT,ambient);
			m->Get(AI_MATKEY_COLOR_SPECULAR,specular);
			mat.diffuse = getColor3(diffuse);
			mat.ambient = getColor3(ambient);
			mat.specular = getColor3(specular);
			m->Get(AI_MATKEY_REFRACTI, mat.refractiveIndex);
		}
	}

	Camera Scene::loadCamera(aiCamera* cam){
		Camera c;
		c.aspectRatio = cam->mAspect;
		c.horizontalFOV = cam->mHorizontalFOV;
		aiVector3D eye = cam->mPosition;
		aiVector3D up = cam->mUp;
		aiVector3D center = cam->mLookAt;
		c.globalTransform = math::lookAt(getTvec3(eye), getTvec3(center), getTvec3(up));
		return c;
	}

	Light** Scene::loadLights(const aiScene* scene){
		numLights = scene->mNumLights;
		Light** ls = new Light*[numLights];
		for(int i = 0; i < numLights; i++){
			aiLight* currentLight = scene->mLights[i];
			Light* l;

			switch(currentLight->mType){
				case aiLightSource_DIRECTIONAL:
					l = (Light*)new DirectionalLight;
					((DirectionalLight*)l)->direction = getVec3(currentLight->mDirection);
					break;
				case aiLightSource_SPOT:
					l = (Light*)new SpotLight;
					((SpotLight*)l)->direction = getVec3(currentLight->mDirection);
					((SpotLight*)l)->innerConeAngle = currentLight->mAngleInnerCone;
					((SpotLight*)l)->outerConeAngle = currentLight->mAngleOuterCone;
					break;
				default:
					l = (Light*)new PointLight;
					break;
			}

			l->attConstant = currentLight->mAttenuationConstant;
			l->attLinear = currentLight->mAttenuationLinear;
			l->attQuadratic = currentLight->mAttenuationQuadratic;
			l->ambient = getColor3(currentLight->mColorAmbient);
			l->diffuse = getColor3(currentLight->mColorDiffuse);
			l->specular = getColor3(currentLight->mColorSpecular);
			l->position = getVec3(currentLight->mPosition);

			ls[i] = l;
		}
		return ls;
	}

	Object* Scene::loadObject(aiNode* node, Object* parent){
		Object* obj = new Object;
		obj->parent = parent;
		getMathMatrix(node->mTransformation, obj->localTransform);

		//if has parent, get transform
		if(parent){
			obj->globalTransform = obj->localTransform * parent->globalTransform;
			obj->globalInverseTransform = inverse(obj->globalTransform);
		}

		if(node->mNumMeshes <= 1){ //Only one mesh
			if(node->mNumMeshes > 0){
				obj->meshIndex = node->mMeshes[0];
			}
			else{
				obj->meshIndex = -1;
			}

			obj->numChildren = node->mNumChildren;
			obj->children = new Object*[node->mNumChildren];

			for(unsigned int i = 0; i < node->mNumChildren; i++){
				obj->children[i] = loadObject(node->mChildren[i], obj);
			}
		}
		else{ //More than one mesh
			obj->meshIndex = -1;
			obj->numChildren = node->mNumChildren + node->mNumMeshes;

			for(unsigned int i = 0; i < node->mNumMeshes; i++){
				Object* child = new Object;
				child->numChildren = 0;
				child->parent = obj;
				child->meshIndex = node->mMeshes[i];
				child->globalTransform = obj->globalTransform;
				child->globalInverseTransform = obj->globalInverseTransform;
			}

			for(unsigned int i = 0; i < node->mNumChildren; i++){
				obj->children[i+node->mNumMeshes] = loadObject(node->mChildren[i], obj);
			}
		}

		return obj;
	}
}