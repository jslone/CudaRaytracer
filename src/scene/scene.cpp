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
	}
}