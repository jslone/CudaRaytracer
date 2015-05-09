#ifndef _BIH_H_
#define _BIH_H_

#include <vector>

#include "math/math.h"
#include "geometry/geometry.h"
#include "vector.h"

namespace acr
{
	template<typename T, size_t MAX_DEPTH = 5>
	class BIH
	{
	public:
		BIH(const std::vector<T> &objs);

		void intersect(const Ray &r, HitInfo &info);

		const static size_t MAX_SIZE = (1 << (MAX_DEPTH - 1));
	private:

		class Node
		{
			uint32_t start, end;
			float left, right;
			uint8_t axis;
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

		void sift(std::vector<Node> tree, int index, const BoundingBox &bb, std::vector<T> objs, size_t start, size_t end)
		{
			tree[index].start = start;
			tree[index].end = end;

			if (index >= MAX_SIZE)
			{
				return;
			}
			if (start == end)
			{
				// zero out bb
				
				return;
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
			int i = start;
			int j = end;
			while (i != j)
			{
				const BoundingBox bb = objs[i].boundingBox;

			}
		}

		vector<Node> tree;
		vector<T> objs;
	};
}

#endif //_BIH_H_