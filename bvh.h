#pragma once

#include "maths.h"
#include "intersection.h"

#include <vector>
#include <algorithm>
#include <cassert>

struct BVHNode
{
	Bounds bounds;
		
	// for leaf nodes these store the range into the items array
	int leftIndex;
	int rightIndex : 31;

	bool leaf : 1;
};

struct BVH
{
	BVHNode* nodes;
	int numNodes;
};

struct BVHBuilder
{
	BVHBuilder(int maxItemsPerLeaf=1) : maxItemsPerLeaf(maxItemsPerLeaf) {}

	BVH Build(const Bounds* items, int n)
	{
		nodes.resize(2*n);
		usedNodes = 0;
	
		bounds.assign(items, items+n);

		// create indices array
		indices.resize(n);
		for (int i=0; i < n; ++i)
			indices[i] = i;

		BuildRecursive(0, n);

		// copy nodes to a new array and return
		BVHNode* nodesCopy = new BVHNode[usedNodes];
		memcpy(nodesCopy, &nodes[0], usedNodes*sizeof(BVHNode));

		BVH bvh;
		bvh.nodes = nodesCopy;
		bvh.numNodes = usedNodes;

		return bvh;
	}

	std::vector<BVHNode> nodes;
	int usedNodes;

	std::vector<Bounds> bounds;
	std::vector<int> indices;

	int maxItemsPerLeaf;

private:

	Bounds CalcBounds(const int* indices, int n)
	{
		Bounds u;

		for (int i=0; i < n; ++i)
			u = Union(u, bounds[indices[i]]);

		return u;
	}

	int LongestAxis(const Vec3& v)
	{
		if (v.x > v.y && v.x > v.z) 
			return 0;
		if (v.y > v.z)
			return 1;
		else
			return 2;
	}
		
	struct PartitionMidPointPredictate
	{
		PartitionMidPointPredictate(const Bounds* bounds, int a, Real m) : bounds(bounds), axis(a), mid(m) {}

		bool operator()(int index) const 
		{
			return bounds[index].GetCenter()[axis] <= mid;
		}

		const Bounds* bounds;
		int axis;
		Real mid;
	};


	int PartitionObjectsMidPoint(int start, int end, Bounds rangeBounds)
	{
		assert(end-start >= 2);

		Vec3 edges = rangeBounds.GetEdges();
		Vec3 center = rangeBounds.GetCenter();

		int longestAxis = LongestAxis(edges);

		Real mid = center[longestAxis];

	
		int* upper = std::partition(&indices[0]+start, &indices[0]+end, PartitionMidPointPredictate(&bounds[0], longestAxis, mid));

		int k = upper-&indices[0];

		return k;
	}

	struct PartitionMedianPredicate
		{
			PartitionMedianPredicate(const Bounds* bounds, int a) : bounds(bounds), axis(a) {}

			bool operator()(int a, int b) const
			{
				return bounds[a].GetCenter()[axis] < bounds[b].GetCenter()[axis];
			}

			const Bounds* bounds;
			int axis;
		};

	int PartitionObjectsMedian(int start, int end, Bounds rangeBounds)
	{
		assert(end-start >= 2);

		Vec3 edges = rangeBounds.GetEdges();

		int longestAxis = LongestAxis(edges);

		const int k = (start+end)/2;

		std::nth_element(&indices[start], &indices[k], &indices[end], PartitionMedianPredicate(&bounds[0], longestAxis));

		return k;
	}	

	int AddNode()
	{
		assert(usedNodes < nodes.size());

		int index = usedNodes;
		++usedNodes;

		return index;
	}

	// returns the index of the node created for this range
	int BuildRecursive(int start, int end)
	{
		assert(start < end);

		const int n = end-start;
		const int nodeIndex = AddNode();

		BVHNode node;
		node.bounds = CalcBounds(&indices[start], end-start);
		
		if (n <= maxItemsPerLeaf)
		{
			node.leaf = true;
			node.leftIndex = indices[start];			
		}
		else
		{
			//const int split = PartitionObjectsMidPoint(start, end, node.bounds);
			int split = PartitionObjectsMedian(start, end, node.bounds);

			if (split == start || split == end)
			{
				/*
				// end splitting early if partitioning fails (only applies to midpoint, not object splitting)
				node.leaf = true;
				node.leftIndex = start;
				node.rightIndex = end;
				*/

				// split at mid point
				assert(0);
				printf("hhow ?");
			}

			

			node.leaf = false;
			node.leftIndex = BuildRecursive(start, split);
			node.rightIndex = BuildRecursive(split, end);
		}

		// output node
		nodes[nodeIndex] = node;

		return nodeIndex;
	}
};


//-----------------------
// Templated query methods

template <typename Func>
CUDA_CALLABLE void QueryRay(const BVHNode* root, Func& f, const Vec3& start, const Vec3& dir)
{
	const Vec3 rcpDir(1.0f/dir.x, 1.0f/dir.y, 1.0f/dir.z);

	const BVHNode* stack[64];
	stack[0] = root;

	int count = 1;

	while (count)
	{
		const BVHNode* n = stack[--count];

		float t;
		//if (IntersectRayAABB(start, dir, n->bounds.lower, n->bounds.upper, t, NULL))
			
		if (IntersectRayAABBFast(start, rcpDir, n->bounds.lower, n->bounds.upper, t))
		{
			if (n->leaf)
			{	
				f(n->leftIndex);
			}
			else
			{
				stack[count++] = &root[n->leftIndex];
				stack[count++] = &root[n->rightIndex];
			}
		}
	}		
}
