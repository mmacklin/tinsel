#pragma once

#include "maths.h"
#include "intersection.h"

#include <vector>
#include <algorithm>
#include <cassert>

class BVH
{
public:

	struct Node
	{
		Bounds bounds;
		
		// for leaf nodes these store the range into the items array
		int leftIndex;
		int rightIndex;

		bool leaf;
	};

	BVH(int maxItemsPerLeaf=1) : maxItemsPerLeaf(maxItemsPerLeaf) {}

	void Build(const Bounds* items, int n)
	{
		nodes.resize(2*n);
		usedNodes = 0;
	
		bounds.assign(items, items+n);

		// create indices array
		indices.resize(n);
		for (int i=0; i < n; ++i)
			indices[i] = i;

		BuildRecursive(0, n);

		// trim arrays
		nodes.resize(usedNodes);
	}

	// calculate Morton codes
	struct KeyIndexPair
	{
		int key;
		int index;

		inline bool operator < (const KeyIndexPair& rhs) const { return key < rhs.key; }
	};


	void BuildFast(const Bounds* items, int n)
	{
		nodes.resize(2*n);
		usedNodes = 0;

		bounds.assign(items, items+n);

		std::vector<KeyIndexPair> keys;
		keys.reserve(n);

		Bounds totalBounds;
		for (int i=0; i < n; ++i)
			totalBounds = Union(totalBounds, items[i]);

		// ensure non-zero edge length in all dimensions
		totalBounds.Expand(0.001f);

		Vec3 edges = totalBounds.GetEdges();
		Vec3 invEdges = Vec3(1.0f)/edges;

		for (int i=0; i < n; ++i)
		{
			Vec3 center = items[i].GetCenter();
			Vec3 local = (center-totalBounds.lower)*invEdges;

			KeyIndexPair l;
			l.key = Morton3(local.x, local.y, local.z);
			l.index = i;

			keys.push_back(l);
		}

		// sort by key
		std::sort(keys.begin(), keys.end());

		// copy indices out to indices
		indices.reserve(n);
		for (int i=0; i < n; ++i)
			indices.push_back(keys[i].index);

		BuildRecursiveFast(&keys[0], 0, n);
	}

	int FindSplit(const KeyIndexPair* pairs, int start, int end)
	{
		// handle case where range has the same key
		if (pairs[start].key == pairs[end].key)
			return (start+end)/2;

		// find split point between keys, xor here means all bits 
		// of the result are zero up until the first differing bit
		int commonPrefix = CLZ(pairs[start].key ^ pairs[end-1].key);

		// use binary search to find the point at which this bit changes
		// from zero to a 1		
		const int mask = 1 << (31-commonPrefix);

		while (end-start > 0)
		{
			int index = (start+end)/2;

			if (pairs[index].key&mask)
			{
				end = index;
			}
			else
				start = index+1;
		}

		assert(start == end);

		return start;
	}

	int BuildRecursiveFast(KeyIndexPair* pairs, int start, int end)
	{
		const int n = end-start;
		const int nodeIndex = AddNode();

		Node node;
		node.bounds = CalcBounds(&indices[start], end-start);
		
		if (n <= maxItemsPerLeaf)
		{
			node.leaf = true;
			node.leftIndex = start;
			node.rightIndex = end;
		}
		else
		{
			int split = FindSplit(pairs, start, end);

			if (split == start || split == end)
			{
				// end splitting early if partitioning fails 
				node.leaf = true;
				node.leftIndex = start;
				node.rightIndex = end;
			}
			else
			{
				node.leaf = false;
				node.leftIndex = BuildRecursiveFast(pairs, start, split);
				node.rightIndex = BuildRecursiveFast(pairs, split, end);
			}
		}

		// output node
		nodes[nodeIndex] = node;

		return nodeIndex;
	}

	template <typename T>
	void QueryBounds(const T& b, std::vector<int>& items, bool precise=false)
	{
		if (nodes.empty())
			return;

		Node* stack[64];
		stack[0] = &nodes[0];

		int count = 1;

		while (count)
		{
			Node* n = stack[--count];

			if (n->bounds.Overlaps(b))
			{
				if (n->leaf)
				{	
					for (int i=n->leftIndex; i < n->rightIndex; ++i)
					{
						if (precise)
						{
							// test individual items
							if (bounds[indices[i]].Overlaps(b))
								items.push_back(indices[i]);
						}
						else
						{
							// push all items in a leaf
							items.push_back(indices[i]);
						}
					}
				}
				else
				{
					stack[count++] = &nodes[n->leftIndex];
					stack[count++] = &nodes[n->rightIndex];					
				}
			}
		}
	}

	
	template <typename T>
	void QueryBoundsSlow(const T& b, std::vector<int>& items)
	{
		for (int i=0; i < indices.size(); ++i)
		{
			if (bounds[indices[i]].Overlaps(b))
				items.push_back(indices[i]);
		}
	}

	template <typename Func>
	void QueryRay(Func& f, const Vec3& start, const Vec3& dir)
	{
		if (nodes.empty())
			return;

		const Vec3 rcpDir(1.0f/dir.x, 1.0f/dir.y, 1.0f/dir.z);

		Node* stack[64];
		stack[0] = &nodes[0];

		int count = 1;

		while (count)
		{
			Node* n = stack[--count];

			float t;
			//if (IntersectRayAABB(start, dir, n->bounds.lower, n->bounds.upper, t, NULL))
			
			if (IntersectRayAABBOmpf(start, rcpDir, n->bounds.lower, n->bounds.upper, t))
			{
				if (n->leaf)
				{	
					for (int i=n->leftIndex; i < n->rightIndex; ++i)
					{
						f(indices[i]);
					}
				}
				else
				{
					stack[count++] = &nodes[n->leftIndex];
					stack[count++] = &nodes[n->rightIndex];
				}
			}
		}		
	}

	std::vector<Node> nodes;
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

	int LongestAxis(const Vec2& v)
	{
		if (v.x > v.y)
			return 0;
		else
			return 1;
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

		Node node;
		node.bounds = CalcBounds(&indices[start], end-start);
		
		if (n <= maxItemsPerLeaf)
		{
			node.leaf = true;
			node.leftIndex = start;
			node.rightIndex = end;
		}
		else
		{
			//const int split = PartitionObjectsMidPoint(start, end, node.bounds);
			const int split = PartitionObjectsMedian(start, end, node.bounds);

			if (split == start || split == end)
			{
				// end splitting early if partitioning fails (only applies to midpoint, not object splitting)
				node.leaf = true;
				node.leftIndex = start;
				node.rightIndex = end;
			}
			else
			{
				node.leaf = false;
				node.leftIndex = BuildRecursive(start, split);
				node.rightIndex = BuildRecursive(split, end);
			}
		}

		// output node
		nodes[nodeIndex] = node;

		return nodeIndex;
	}
};




