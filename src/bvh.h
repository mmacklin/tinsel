#pragma once

#include "maths.h"

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

static_assert(sizeof(BVHNode) == 32, "Error BVHNode size larger than expected");

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

		std::nth_element(&indices[0]+start, &indices[0]+k, &indices[0]+end, PartitionMedianPredicate(&bounds[0], longestAxis));

		return k;
	}	

	float Area(const Bounds& b)
	{
		Vec3 edges = b.GetEdges();

		return 2.0f*(edges.x*edges.y + edges.x*edges.z + edges.y*edges.z);

	}

	int PartitionObjectsSAH(int start, int end, Bounds rangeBounds)
	{
		assert(end-start >= 2);

		int n = end-start;
		Vec3 edges = rangeBounds.GetEdges();

		int longestAxis = LongestAxis(edges);

		// sort along longest axis
		std::sort(&indices[0]+start, &indices[0]+end, PartitionMedianPredicate(&bounds[0], longestAxis));

		// total area for range from [0, split]
		std::vector<float> leftAreas(n);
		// total area for range from (split, end]
		std::vector<float> rightAreas(n);

		Bounds left;
		Bounds right;

		// build cumulative bounds and area from left and right
		for (int i=0; i < n; ++i)
		{
			left = Union(left, bounds[indices[start+i]]);
			right = Union(right, bounds[indices[end-i-1]]);

			leftAreas[i] = Area(left);
			rightAreas[n-i-1] = Area(right);
		}

		float invTotalArea = 1.0f/Area(rangeBounds);

		// find split point i that minimizes area(left[i]) + area(right[i])
		int minSplit = 0;
		float minCost = FLT_MAX;

		for (int i=0; i < n; ++i)
		{
			float pBelow = leftAreas[i]*invTotalArea;
			float pAbove = rightAreas[i]*invTotalArea;

			float cost = pBelow*i + pAbove*(n-i);

			if (cost < minCost)
			{
				minCost = cost;
				minSplit = i;
			}
		}

		return start + minSplit + 1;
	}	

	int AddNode()
	{
		assert(usedNodes < nodes.size());

		int index = usedNodes;
		++usedNodes;

		return index;
	}

	// returns the index of the node created for this range [start, end)
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
			//int split = PartitionObjectsMedian(start, end, node.bounds);
			int split = PartitionObjectsSAH(start, end, node.bounds);

			if (split == start || split == end)
			{
				// partitioning failed, split down the middle
				split = (start+end)/2;
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

