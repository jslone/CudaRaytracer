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
			: objs(hObjs)
		{
			thrust::host_vector<Node> hTree(MAX_SIZE);

			sift(hTree, 0, bb, hObjs, 0, hObjs.size());

			tree = vector<Node>(hTree);
		}

		void intersect(const Ray &r, HitInfo &info);

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
			return &tree[getLeftChildIdx(tree.position(n))];
		}

		Node* getRightChild(const Node *n)
		{
			return &tree[getRightChildIdx(tree.position(n))];
		}

		bool sift(thrust::host_vector<Node> tree, int index, const BoundingBox &bb, thrust::host_vector<T> objs, size_t start, size_t end)
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

			if (sift(tree, getLeftChildIdx(index), lBB, objs, start, i) ||
				sift(tree, getRightChildIdx(index), rBB, objs, i, end))
			{
				tree[index].isLeaf = true;
			}

			return false;
		}

		vector<Node> tree;
		vector<T> objs;
	};
}

#endif //_BIH_H_