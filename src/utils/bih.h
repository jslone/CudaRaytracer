#ifndef _BIH_H_
#define _BIH_H_

#include <limits>
#include <thrust/host_vector.h>

#include "math/math.h"
#include "geometry/geometry.h"
#include "vector.h"

namespace acr
{
	struct Path
	{
		__device__ __host__
		Path() : path(0), length(0) {}
		uint64_t path;
		uint64_t length;

		__device__ __host__ inline
		void append(uint64_t flag, uint64_t size)
		{
			length += size;
			if (length <= sizeof(path))
				path |= flag << (sizeof(path) - length);
		}
	};

	template<typename T, size_t MAX_DEPTH = 6>
	class BIH
	{
	public:
		BIH() = default;
		BIH(const BIH &) = default;

		BIH(thrust::host_vector<T> &hObjs, const BoundingBox &bb, void *data)
			: boundingBox(bb)
		{
			sift(0, boundingBox, hObjs, 0, hObjs.size(), data);
			objs = vector<T>(hObjs);
			std::cout << *this << std::endl;
		}

		__device__
		bool intersect(const Ray &r, HitInfo &info, void *data, Path &path)
		{
			return treeIntersect(r, info, data, boundingBox, path);
		}

		const static size_t MAX_SIZE = (1 << MAX_DEPTH) - 1;
	private:

		struct Node
		{
			uint32_t start, end;
			float left, right;
			uint8_t axis : 2;
			uint8_t isLeaf : 1;
		};

		enum AXIS : uint8_t
		{
			X = 0, Y, Z
		};

		__device__ __host__ inline
		size_t getLeftChildIdx(size_t idx)
		{
			return 2 * idx + 1;
		}

		__device__ __host__ inline
		size_t getRightChildIdx(size_t idx)
		{
			return 2 * idx + 2;
		}

		__device__ __host__ inline
		Node* getLeftChild(const Node *n)
		{
			return &tree[getLeftChildIdx(tree - n)];
		}

		__device__ __host__ inline
		Node* getRightChild(const Node *n)
		{
			return &tree[getRightChildIdx(tree - n)];
		}

		bool sift(int index, const BoundingBox &bb, thrust::host_vector<T> &hObjs, size_t start, size_t end, void *data)
		{	
			if (index >= MAX_SIZE)
			{
				return true;
			}

			tree[index].start = start;
			tree[index].end = end;

			if (start >= end)
			{
				tree[index].isLeaf = true;
				return true;
			}

			// pick axis
			float len = -1;
			uint8_t axis = 0;
			for (int i = 0; i < 3; i++)
			{
				float axisLen = bb.max[i] - bb.min[i];
				if (axisLen > len)
				{
					len = axisLen;
					axis = i;
				}
			}

			// pick pivot
			float pivot = bb.min[axis] + len / 2;

			// pivot
			BoundingBox lBB;
			BoundingBox rBB;

			lBB.min = bb.max;
			lBB.max = bb.min;

			rBB = lBB;

			int i = start;
			int j = end - 1;

			while (i <= j)
			{
				const BoundingBox bb = hObjs[i].getBoundingBox(data);

				float pos = hObjs[i].getCentroid(data)[axis];

				if (pos <= pivot)
				{
					lBB.min = math::min(lBB.min, bb.min);
					lBB.max = math::max(lBB.max, bb.max);
					i++;
				}
				else
				{
					rBB.min = math::min(rBB.min, bb.min);
					rBB.max = math::max(rBB.max, bb.max);
					if (i != j)
					{
						std::swap(hObjs[i], hObjs[j]);
					}
					j--;
				}
			}

			// Recurse on left and right
			
			tree[index].left = lBB.max[axis];
			tree[index].right = rBB.min[axis];
			tree[index].axis = axis;

			bool left = sift(getLeftChildIdx(index), lBB, hObjs, start, i, data);
			bool right = sift(getRightChildIdx(index), rBB, hObjs, i, end, data);
			tree[index].isLeaf = left || right;
			return false;
		}

		__device__
		bool treeIntersect(const Ray &r, HitInfo &info, const void *data, const BoundingBox &bb, Path &path)
		{
			bool intersected = false;
			
			int treeIdx[MAX_DEPTH+2];
			BoundingBox treeBB[MAX_DEPTH+2];
			int depth = 0;

			treeIdx[0] = 0;
			treeBB[0] = bb;

			uint32_t bitsAppended = 0;

			BoundingBox::Args bbArgs;
			
			bbArgs.invD = 1.0f / r.d;
			bbArgs.sign.x = bbArgs.invD.x < 0;
			bbArgs.sign.y = bbArgs.invD.y < 0;
			bbArgs.sign.z = bbArgs.invD.z < 0;

			while (depth >= 0)
			{
				int i = treeIdx[depth];
				const BoundingBox &bb = treeBB[depth];
				const Node &n = tree[i];

				// intersect objects
				if (n.isLeaf)
				{
					for (int i = n.start; i < n.end; i++)
					{
						intersected |= objs[i].intersect(r, info, data);
					}
				}
				// intersect children
				else
				{
					BoundingBox lBB = bb;
					BoundingBox rBB = bb;

					lBB.max[n.axis] = n.left;
					rBB.min[n.axis] = n.right;

					bool hitL = lBB.intersect(r, info, bbArgs);
					bool hitR = rBB.intersect(r, info, bbArgs);

					int lIdx = getLeftChildIdx(i);
					int rIdx = getRightChildIdx(i);

					if (hitR)
					{
						treeIdx[depth] = rIdx;
						treeBB[depth] = rBB;
						depth++;
					}
					if (hitL)
					{
						treeIdx[depth] = lIdx;
						treeBB[depth] = lBB;
						depth++;
					}
					
					// left got more than right
					bool lBigger = (tree[lIdx].end - tree[lIdx].start) > (tree[rIdx].end - tree[rIdx].start);
					
					// append to msb 00 for neither hit, 11 for both, 10 for bigger hit, 01 for smaller hit
					path.append(hitL << (lBigger) | hitR << (!lBigger), 2);
					bitsAppended += 2;
				}
				depth--;
			}
			uint32_t padding = 2 * MAX_SIZE - bitsAppended;
			if (padding > 0)
			{
				path.append(0, padding);
			}
			return intersected;
		}

		std::ostream& outputNode(std::ostream &os, uint32_t index, uint32_t depth)
		{
			os << std::string(depth, '\t') << index << ": [" << tree[index].start << ":" << tree[index].end << "]" << std::endl;
			if (!tree[index].isLeaf)
			{
				outputNode(outputNode(os, getLeftChildIdx(index), depth + 1), getRightChildIdx(index), depth + 1);
			}
			return os;
		}

		friend std::ostream& operator<<(std::ostream& os, BIH<T,MAX_DEPTH> &bih)
		{
			return bih.outputNode(os, 0, 0);
		}

		BoundingBox boundingBox;
		Node tree[MAX_SIZE];
		vector<T> objs;
	};
}

#endif //_BIH_H_