#ifndef _BIH_H_
#define _BIH_H_

#include <limits>
#include <thrust/host_vector.h>

#include "math/math.h"
#include "geometry/geometry.h"
#include "vector.h"

namespace acr
{
	template<typename T, size_t MAX_DEPTH = 5>
	class BIH
	{
	public:
		BIH() = default;
		BIH(const BIH &) = default;

		BIH(const thrust::host_vector<T> &hObjs, const BoundingBox &bb)
			: boundingBox(bb)
			, objs(hObjs)
		{
			sift(0, boundingBox, hObjs, 0, hObjs.size());
		}

		bool intersect(const Ray &r, HitInfo &info, void *data)
		{
			return treeIntersect(r, info, data, boundingBox);
		}

		const static size_t MAX_SIZE = (1 << (MAX_DEPTH - 1));
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

		size_t getLeftChildIdx(size_t idx)
		{
			return 2 * idx + 1;
		}

		size_t getRightChildIdx(size_t idx)
		{
			return 2 * idx + 2;
		}

		Node* getLeftChild(const Node *n)
		{
			return &tree[getLeftChildIdx(tree - n)];
		}

		Node* getRightChild(const Node *n)
		{
			return &tree[getRightChildIdx(tree - n)];
		}

		bool sift(int index, const BoundingBox &bb, thrust::host_vector<T> objs, size_t start, size_t end)
		{
			tree[index].start = start;
			tree[index].end = end;

			if (index >= MAX_SIZE || start >= end)
			{
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

			float minL = std::numeric_limits<float>::infinity();
			float maxL = -std::numeric_limits<float>::infinity();

			float minR = std::numeric_limits<float>::infinity();
			float maxR = -std::numeric_limits<float>::infinity();

			int i = start;
			int j = end;

			while (i <= j)
			{
				const BoundingBox bb = objs[i].boundingBox;

				float pos = objs[i].centroid[axis];
				float minB = bb.min[axis];
				float maxB = bb.max[axis];

				if (pos <= pivot)
				{
					minL = math::min(minL, minB);
					maxL = math::max(maxL, maxB);
					i++;
				}
				else
				{
					minL = math::min(minL, minB);
					maxL = math::max(maxL, maxB);
					if (i != j)
					{
						std::swap(objs[i], objs[j]);
					}
					j--;
				}
			}

			// Recurse on left and right
			BoundingBox lBB = bb;
			BoundingBox rBB = bb;

			lBB.min[axis] = minL;
			lBB.max[axis] = maxL;

			rBB.min[axis] = minR;
			rBB.max[axis] = maxR;

			tree[index].left = maxL;
			tree[index].right = minR;

			if (sift(getLeftChildIdx(index), lBB, objs, start, i) ||
				sift(getRightChildIdx(index), rBB, objs, i, end))
			{
				tree[index].isLeaf = true;
			}

			return false;
		}

		bool treeIntersect(const Ray &r, HitInfo &info, const void *data, const BoundingBox &bb)
		{
			bool intersected = false;
			
			int treeIdx[MAX_DEPTH];
			BoundingBox treeBB[MAX_DEPTH];
			int depth = 0;

			treeIdx[0] = 0;
			treeBB[0] = bb;

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

					bool hitL = lBB.intersect(r, info, args);
					bool hitR = rBB.intersect(r, info, args);

					if (hitR)
					{
						treeIdx[depth] = getRightChildIdx(i);
						treeBB[depth] = rBB;
						depth++;
					}
					if (hitL)
					{
						treeIdx[depth] = getLeftChildIdx(i);
						treeBB[depth] = lBB;
						depth++;
					}
				}
				depth--;
			}

			return intersected;
		}

		BoundingBox boundingBox;
		Node tree[MAX_SIZE];
		vector<T> objs;
	};
}

#endif //_BIH_H_