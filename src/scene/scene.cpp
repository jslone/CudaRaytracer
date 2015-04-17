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

	void loadScene(aiScene* scene){
		//scene->mCameras[]		scene->mNumCameras
		//scene->mLights[]		scene->mNumLights
		//scene->mMaterials[]	scene->mNumMaterials
		//scene->mMeshes[]		scene->mNumMeshes
		//scene->mTextures[]	scene->mNumTextures

		//scene->mRootNode		root node of hierarchy
		//scene->mRootNode.mChildren[]			scene->mRootNode.mNumChildren
		//scene->mRootNode.mMeshes[]			scene->mRootNode.mNumMeshes
		//scene->mRootNode.mParent				#aiNode*
		//scene->mRootNode.mTransformation		#aiMatrix4x4
		//scene->mRootNode.mName 				#aiString

		//THINGS TO DO:

		//Load camera
		//Load lights
		//Load materials
		//Load textures

		//Load meshes

		//Load hierarchies
		loadNode(scene->mRootNode, NULL);
	}

	math::mat4& getMathMatrix(aiMatrix4x4 aiMatrix){
		math::mat4 mat;
		for(int i = 0; i < 4; i++){
			for(int j = 0; j < 4; j++){
				mat[i][j] = aiMatrix[j][i];
			}
		}
		return mat;
	}

	Object* loadNode(aiNode* node, Object* parent){
		Object* obj = new Object;
		obj->parent = parent;
		obj->localTransform = getMathMatrix(node->mTransformation);

		//If has parent, get transform
		if(parent){
			obj->globalTransform = obj->localTransform * parent->globalTransform;
			obj->globalInverseTransform = inverse(obj->globalTransform);
		}

		//Only one mesh
		if(node->mNumMeshes <= 1){
			if(node->mNumMeshes > 0){
				obj->meshIndex = node->mMeshes[0]; //Get actual mesh
			}
			else{
				obj->meshIndex = -1;
			}

			obj->numChildren = node->mNumChildren;
			obj->children = new Object[mNumChildren];

			for(int i = 0; i < node->mNumChildren; i++){
				obj->children[i] = loadNode(node->mChildren[i], obj);
			}
		}
		//More than one mesh
		else{
			obj->meshIndex = -1;
			obj->numChildren = node->mNumChildren + node->mNumMeshes;

			for(int i = 0; i < node->mNumMeshes; i++){
				Object* child = new Object;
				child->numChildren = 0;
				child->parent = obj;
				child->meshIndex = node->mMeshes[i]; //Get actual mesh
				child->globalTransform = obj->globalTransform;
				child->globalInverseTransform = obj->globalInverseTransform;
			}

			for(int i = 0; i < node->mNumChildren; i++){
				obj->children[i+mNumMeshes] = loadNode(node->mChildren[i], obj);
			}
		}

		return obj;
	}
}