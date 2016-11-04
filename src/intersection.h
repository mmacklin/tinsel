#pragma once

#include "maths.h"
#include "mesh.h"
#include "scene.h"
#include "sampler.h"

template <typename T>
CUDA_CALLABLE CUDA_CALLABLE inline void Sort2(T& a, T& b)
{
	if (b < a)
		Swap(a, b);
}

template <typename T>
CUDA_CALLABLE CUDA_CALLABLE inline void Sort3(T& a, T& b, T& c)
{
	if (b < a)
		Swap(a, b);
	if (c < b)
		Swap(b, c);
	if (b < a)
		Swap(a, b);

	assert(a <= b);
	assert(b <= c);
}

template <typename T>
CUDA_CALLABLE CUDA_CALLABLE inline bool SolveQuadratic(T a, T b, T c, T& minT, T& maxT)
{
	if (a == 0.0f && b == 0.0f)
	{
		minT = maxT = 0.0f;
		return true;
	}

	T discriminant = b*b - T(4.0)*a*c;

	if (discriminant < 0.0f)
	{
		return false;
	}

	// numerical receipes 5.6 (this method ensures numerical accuracy is preserved)
	T t = T(-0.5) * (b + Sign(b)*Sqrt(discriminant));
	minT = t / a;
	maxT = c / t;

	Sort2(minT, maxT);

	return true;
}




// alternative ray sphere intersect, returns closest and furthest t values
CUDA_CALLABLE inline bool IntersectRaySphere(const Vec3& sphereOrigin, float sphereRadius, const Vec3& rayOrigin, const Vec3& rayDir, float& minT, float &maxT, Vec3* hitNormal=NULL)
{
	Vec3 q = rayOrigin-sphereOrigin;

	float a = 1.0f;
	float b = 2.0f*Dot(q, rayDir);
	float c = Dot(q, q)-(sphereRadius*sphereRadius);

	bool r = SolveQuadratic(a, b, c, minT, maxT);

	if (minT < 0.0f && maxT < 0.0f)
		return false;
	
	if (minT < 0.0f && maxT > 0.0f)
		minT = maxT;
		

	// calculate the normal of the closest hit
	if (hitNormal && r)
	{
		*hitNormal = Normalize((rayOrigin+rayDir*minT)-sphereOrigin);
	}

	return r;
}

CUDA_CALLABLE inline bool IntersectRayPlane(const Vec3& p, const Vec3& dir, const Vec4& plane, float& t)
{
    float d = Dot(plane, Vec4(dir, 0.0f));
    
    if (d == 0.0f)
    {
        return false;
    }
	else
    {
        t = -Dot(plane, Vec4(p, 1.0f)) / d;
    }

	return (t > 0.0f);	
}

CUDA_CALLABLE inline bool IntersectLineSegmentPlane(const Vec3& start, const Vec3& end, const Vec4& plane, Vec3& out)
{
	Vec3 u(end - start);

	float dist = -Dot(plane, Vec4(start, 1.0f)) / Dot(plane, Vec4(u, 0.0f));

	if (dist > 0.0f && dist < 1.0f)
	{
		out = (start + u * dist);
		return true;
	}
	else
		return false;
}

// Moller and Trumbore's method
CUDA_CALLABLE inline bool IntersectRayTriTwoSided(const Vec3& p, const Vec3& dir, const Vec3& a, const Vec3& b, const Vec3& c, float& t, float& u, float& v, float& w, float& sign, Vec3* normal)
{
    Vec3 ab = b - a;
    Vec3 ac = c - a;
    Vec3 n = Cross(ab, ac);

    float d = Dot(-dir, n);
    float ood = 1.0f / d; // No need to check for division by zero here as infinity aritmetic will save us...
    Vec3 ap = p - a;

    t = Dot(ap, n) * ood;
    if (t < 0.0f)
        return false;

    Vec3 e = Cross(-dir, ap);
    v = Dot(ac, e) * ood;
    if (v < 0.0f || v > 1.0f) // ...here...
        return false;
    w = -Dot(ab, e) * ood;
    if (w < 0.0f || v + w > 1.0f) // ...and here
        return false;

    u = 1.0f - v - w;
    if (normal)
        *normal = n;
	sign = d;

    return true;
}



// mostly taken from Real Time Collision Detection - p192
CUDA_CALLABLE inline bool IntersectRayTri(const Vec3& p, const Vec3& dir, const Vec3& a, const Vec3& b, const Vec3& c,  float& t, float& u, float& v, float& w, Vec3* normal)
{
	const Vec3 ab = b-a;
	const Vec3 ac = c-a;

	// calculate normal
	Vec3 n = Cross(ab, ac);

	// need to solve a system of three equations to give t, u, v
	float d = Dot(-dir, n);

	// if dir is parallel to triangle plane or points away from triangle 
	if (d <= 0.0f)
        return false;

	Vec3 ap = p-a;
	t = Dot(ap, n);

	// ignores tris behind 
	if (t < 0.0f)
		return false;

	// compute barycentric coordinates
	Vec3 e = Cross(-dir, ap);
	v = Dot(ac, e);
	if (v < 0.0f || v > d) return false;

	w = -Dot(ab, e);
	if (w < 0.0f || v + w > d) return false;

	float ood = 1.0f / d;
	t *= ood;
	v *= ood;
	w *= ood;
	u = 1.0f-v-w;

	// optionally write out normal (todo: this branch is a performance concern, should probably remove)
	if (normal)
		*normal = Normalize(n);

	return true;
}
/*
// mostly taken from Real Time Collision Detection - p192
CUDA_CALLABLE CUDA_CALLABLE inline bool IntersectSegmentTri(const Vec3& p, const Vec3& q, const Vec3& a, const Vec3& b, const Vec3& c,  float& t, float& u, float& v, float& w, Vec3* normal, float expand)
{
	const Vec3 ab = b-a;
	const Vec3 ac = c-a;
	const Vec3 qp = p-q;

	// calculate normal
	Vec3 n = Cross(ab, ac);

	// need to solve a system of three equations to give t, u, v
	float d = Dot(qp, n);

	// if dir is parallel to triangle plane or points away from triangle 
	if (d <= 0.0f)
        return false;

	Vec3 ap = p-a;
	t = Dot(ap, n);

	// ignores tris behind 
	if (t < 0.0f)
		return false;

	// ignores tris beyond segment
	if (t > d)
		return false;

	// compute barycentric coordinates
	Vec3 e = Cross(qp, ap);
	v = Dot(ac, e);
	if (v < 0.0f || v > d) return false;

	w = -Dot(ab, e);
	if (w < 0.0f || v + w > d) return false;

	float ood = 1.0f / d;
	t *= ood;
	v *= ood;
	w *= ood;
	u = 1.0f-v-w;

	// optionally write out normal (todo: this branch is a performance concern, should probably remove)
	if (normal)
		*normal = n;

	return true;
}
*/

CUDA_CALLABLE CUDA_CALLABLE inline float ScalarTriple(const Vec3& a, const Vec3& b, const Vec3& c) { return Dot(Cross(a, b), c); }

// mostly taken from Real Time Collision Detection - p192
CUDA_CALLABLE CUDA_CALLABLE inline bool IntersectSegmentTri(const Vec3& p, const Vec3& q, const Vec3& a, const Vec3& b, const Vec3& c,  float& t, float& u, float& v, float& w, Vec3* normal, float expand)
{
	const Vec3 pq = q-p;
	const Vec3 pa = a-p;
	const Vec3 pb = b-p;
	const Vec3 pc = c-p;

	u = ScalarTriple(pq, pc, pb);
	if (u < 0.0f) return 0;

	v = ScalarTriple(pq, pa, pc);
	if (v < 0.0f) return 0;

	w = ScalarTriple(pq, pb, pa);
	if (w < 0.0f) return 0;

	return true;
}

// RTCD 5.1.5, page 142
CUDA_CALLABLE CUDA_CALLABLE inline  Vec3 ClosestPointOnTriangle(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3 p)
{
	Vec3 ab = b- a;
	Vec3 ac = c-a;
	Vec3 ap = p-a;
	
	float d1 = Dot(ab, ap);
	float d2 = Dot(ac, ap);
	if (d1 <= 0.0f && d2 <= 0.0f)
		return a;

	Vec3 bp = p-b;
	float d3 = Dot(ab, bp);
	float d4 = Dot(ac, bp);
	if (d3 >= 0.0f && d4 <= d3)
		return b;

	float vc = d1*d4 - d3*d2;
	if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
	{
		float v = d1 / (d1-d3);
		return a + v*ab;
	}

	Vec3 cp =p-c;
	float d5 = Dot(ab, cp);
	float d6 = Dot(ac, cp);
	if (d6 >= 0.0f && d5 <= d6)
		return c;

	float vb = d5*d2 - d1*d6;
	if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
	{
		float w = d2 / (d2 - d6);
		return a + w * ac;
	}

	float va = d3*d6 - d5*d4;
	if (va <= 0.0f && (d4 -d3) >= 0.0f && (d5-d6) >= 0.0f)
	{
		float w = (d4-d3)/((d4-d3) + (d5-d6));
		return b + w * (c-b);
	}

	float denom = 1.0f / (va + vb + vc);
	float v = vb * denom;
	float w = vc * denom;
	return a + ab*v + ac*w;
}


CUDA_CALLABLE CUDA_CALLABLE inline float SqDistPointSegment(Vec3 a, Vec3 b, Vec3 c)
{
	Vec3 ab = b-a, ac=c-a, bc=c-b;
	float e = Dot(ac, ab);

	if (e <= 0.0f)
		return Dot(ac, ac);
	float f = Dot(ab, ab);
	
	if (e >= f)
		return Dot(bc, bc);

	return Dot(ac, ac) - e*e/f;
}

CUDA_CALLABLE CUDA_CALLABLE inline bool PointInTriangle(Vec3 a, Vec3 b, Vec3 c, Vec3 p)
{
	a -= p; b -= p; c-= p;

	/*
	float eps = 0.0f;

	float ab = Dot(a, b);
	float ac = Dot(a, c);
	float bc = Dot(b, c);
	float cc = Dot(c, c);

	if (bc *ac - cc * ab <= eps)
		return false;

	float bb = Dot(b, b);
	if (ab * bc - ac*bb <= eps)
		return false;

	return true;
	*/

	Vec3 u = Cross(b, c);
	Vec3 v = Cross(c, a);

	if (Dot(u, v) <= 0.0f)
		return false;

	Vec3 w = Cross(a, b);

	if (Dot(u, w) <= 0.0f)
		return false;
	
	return true;
}


CUDA_CALLABLE inline float minf(const float a, const float b) { return a < b ? a : b; }
CUDA_CALLABLE inline float maxf(const float a, const float b) { return a > b ? a : b; }

// from Ompf
CUDA_CALLABLE inline bool IntersectRayAABBFast(const Vec3& pos, const Vec3& rcp_dir, const Vec3& min, const Vec3& max, float& t) {
       
    float
    l1	= (min.x - pos.x) * rcp_dir.x,
    l2	= (max.x - pos.x) * rcp_dir.x,
    lmin	= minf(l1,l2),
    lmax	= maxf(l1,l2);

    l1	= (min.y - pos.y) * rcp_dir.y;
    l2	= (max.y - pos.y) * rcp_dir.y;
    lmin	= maxf(minf(l1,l2), lmin);
    lmax	= minf(maxf(l1,l2), lmax);

    l1	= (min.z - pos.z) * rcp_dir.z;
    l2	= (max.z - pos.z) * rcp_dir.z;
    lmin	= maxf(minf(l1,l2), lmin);
    lmax	= minf(maxf(l1,l2), lmax);

    //return ((lmax > 0.f) & (lmax >= lmin));
    //return ((lmax > 0.f) & (lmax > lmin));
    bool hit = ((lmax >= 0.f) & (lmax >= lmin));
    if (hit)
        t = lmin;
    return hit;
}


CUDA_CALLABLE inline bool IntersectRayAABB(const Vec3& start, const Vec3& dir, const Vec3& min, const Vec3& max, float& t, Vec3* normal)
{
	//! calculate candidate plane on each axis
	float tx = -1.0f, ty = -1.0f, tz = -1.0f;
	bool inside = true;
			
	//! use unrolled loops

	//! x
	if (start.x < min.x)
	{
		if (dir.x != 0.0f)
			tx = (min.x-start.x)/dir.x;
		inside = false;
	}
	else if (start.x > max.x)
	{
		if (dir.x != 0.0f)
			tx = (max.x-start.x)/dir.x;
		inside = false;
	}

	//! y
	if (start.y < min.y)
	{
		if (dir.y != 0.0f)
			ty = (min.y-start.y)/dir.y;
		inside = false;
	}
	else if (start.y > max.y)
	{
		if (dir.y != 0.0f)
			ty = (max.y-start.y)/dir.y;
		inside = false;
	}

	//! z
	if (start.z < min.z)
	{
		if (dir.z != 0.0f)
			tz = (min.z-start.z)/dir.z;
		inside = false;
	}
	else if (start.z > max.z)
	{
		if (dir.z != 0.0f)
			tz = (max.z-start.z)/dir.z;
		inside = false;
	}

	//! if point inside all planes
	if (inside)
    {
        t = 0.0f;
		return true;
    }

	//! we now have t values for each of possible intersection planes
	//! find the maximum to get the intersection point
	float tmax = tx;
	int taxis = 0;

	if (ty > tmax)
	{
		tmax = ty;
		taxis = 1;
	}
	if (tz > tmax)
	{
		tmax = tz;
		taxis = 2;
	}

	if (tmax < 0.0f)
		return false;

	//! check that the intersection point lies on the plane we picked
	//! we don't test the axis of closest intersection for precision reasons

	//! no eps for now
	float eps = 0.0f;

	Vec3 hit = start + dir*tmax;

	if ((hit.x < min.x-eps || hit.x > max.x+eps) && taxis != 0)
		return false;
	if ((hit.y < min.y-eps || hit.y > max.y+eps) && taxis != 1)
		return false;
	if ((hit.z < min.z-eps || hit.z > max.z+eps) && taxis != 2)
		return false;

	//! output results
	t = tmax;
			
	return true;
}

// construct a plane equation such that ax + by + cz + dw = 0
CUDA_CALLABLE inline Vec4 PlaneFromPoints(const Vec3& p, const Vec3& q, const Vec3& r)
{
	Vec3 e0 = q-p;
	Vec3 e1 = r-p;

	Vec3 n = SafeNormalize(Cross(e0, e1));
	
	return Vec4(n.x, n.y, n.z, -Dot(p, n));
}

CUDA_CALLABLE inline bool IntersectPlaneAABB(const Vec4& plane, const Vec3& center, const Vec3& extents)
{
	float radius = Abs(extents.x*plane.x) + Abs(extents.y*plane.y) + Abs(extents.z*plane.z);
	float delta = Dot(center, Vec3(plane)) + plane.w;

	return Abs(delta) <= radius;
}


//-----------------------
// Templated query methods

template <typename Func>
CUDA_CALLABLE void QueryRayOld(const BVHNode* root, Func& f, const Vec3& start, const Vec3& dir)
{
	Vec3 rcpDir;
	rcpDir.x = 1.0f/dir.x;
	rcpDir.y = 1.0f/dir.y;
	rcpDir.z = 1.0f/dir.z;

	int stack[32];
	stack[0] = 0;

	int count = 1;

	while (count)
	{
		const BVHNode* n = root + stack[--count];

#if CUDA
		// load using two 128 bit loads
		float4 f4[2];

		const float4* __restrict__ fn = (const float4*)(n);
		f4[0] = fn[0];
		f4[1] = fn[1];

		const BVHNode& node = (const BVHNode&)(f4[0]);
#else
		const BVHNode node = *n;
#endif

		float t;
		//if (IntersectRayAABB(start, dir, n->bounds.lower, n->bounds.upper, t, NULL))			
		if (IntersectRayAABBFast(start, rcpDir, node.bounds.lower, node.bounds.upper, t))
		{
			if (node.leaf)
			{	
				f(node.leftIndex);
			}
			else
			{
				stack[count++] = node.leftIndex;
				stack[count++] = node.rightIndex;
			}
		}
	}		
}

CUDA_CALLABLE inline BVHNode fetchNode(const BVHNode* ptr, int index)
{
#if __CUDA_ARCH__ && USE_TEXTURES
		
		// load using two 128 bit loads
		float4 f4[2];

		f4[0] = tex1Dfetch<float4>((cudaTextureObject_t)ptr, index*2+0);
		f4[1] = tex1Dfetch<float4>((cudaTextureObject_t)ptr, index*2+1);

		return (const BVHNode&)(f4[0]);
#else
		return ptr[index];
#endif
}

template <typename Func>
CUDA_CALLABLE void QueryRay(const BVHNode* root, Func& f, const Vec3& start, const Vec3& dir)
{
	Vec3 rcpDir;
	rcpDir.x = 1.0f/dir.x;
	rcpDir.y = 1.0f/dir.y;
	rcpDir.z = 1.0f/dir.z;

	int stack[32];
	stack[0] = 0;

	int count = 1;

	while(count)
	{
		BVHNode node = fetchNode(root, stack[--count]);

		if (node.leaf)
		{
			f(node.leftIndex);
		}
		else
		{
			// check children
			BVHNode left = fetchNode(root, node.leftIndex);
			BVHNode right = fetchNode(root, node.rightIndex);

			float tLeft = FLT_MAX;
			bool hitLeft = IntersectRayAABBFast(start, rcpDir, left.bounds.lower, left.bounds.upper, tLeft);

			float tRight = FLT_MAX;
			bool hitRight = IntersectRayAABBFast(start, rcpDir, right.bounds.lower, right.bounds.upper, tRight);

			if (hitLeft)
				stack[count++] = node.leftIndex;

			if (hitRight)
				stack[count++] = node.rightIndex;
		}
	}		
}


struct MeshQuery
{
	CUDA_CALLABLE inline MeshQuery(const MeshGeometry& m, const Vec3& origin, const Vec3& dir) : mesh(m), rayOrigin(origin), rayDir(dir), closestT(FLT_MAX) {}
	
	CUDA_CALLABLE inline void operator()(int i)
	{	
		float t, u, v, w;
		Vec3 n;

		int i0 = fetchInt(mesh.indices, i*3+0);
		int i1 = fetchInt(mesh.indices, i*3+1);
		int i2 = fetchInt(mesh.indices, i*3+2);

		const Vec3 a = fetchVec3(mesh.positions, i0);
		const Vec3 b = fetchVec3(mesh.positions, i1);
		const Vec3 c = fetchVec3(mesh.positions, i2);

		float sign;
		//if (IntersectRayTri(rayOrigin, rayDir, a, b, c, t, u, v, w, &n))
		if (IntersectRayTriTwoSided(rayOrigin, rayDir, a, b, c, t, u, v, w, sign, &n))
		{
			if (t > 0.0f && t < closestT)
			{
				closestT = t;
				closestU = u;
				closestV = v;
				closestW = w;

				closestTri = i;
				closestNormal = n*sign;
			}
		}
	}
	
	const MeshGeometry mesh;
	const Vec3 rayOrigin;
	const Vec3 rayDir;
	
	float closestT;
	float closestU;
	float closestV;
	float closestW;

	Vec3 closestNormal;
	int closestTri;
};



CUDA_CALLABLE bool inline IntersectRayMesh(const MeshGeometry& mesh, const Vec3& origin, const Vec3& dir, float tmax, float& t, float& u, float& v, float& w, int& tri, Vec3& triNormal)
{

	MeshQuery query(mesh, origin, dir);

	Vec3 rcpDir;
	rcpDir.x = 1.0f/dir.x;
	rcpDir.y = 1.0f/dir.y;
	rcpDir.z = 1.0f/dir.z;

	int stack[32];
	stack[0] = 0;

	int count = 1;

	while(count)
	{
		BVHNode node = fetchNode(mesh.nodes, stack[--count]);

		if (node.leaf)
		{
			query(node.leftIndex);

			// truncate ray
			tmax = query.closestT;
		}
		else
		{
			// check children
			BVHNode left = fetchNode(mesh.nodes, node.leftIndex);
			BVHNode right = fetchNode(mesh.nodes, node.rightIndex);

			float tLeft;
			bool hitLeft = IntersectRayAABBFast(origin, rcpDir, left.bounds.lower, left.bounds.upper, tLeft) && tLeft < tmax;

			float tRight;
			bool hitRight = IntersectRayAABBFast(origin, rcpDir, right.bounds.lower, right.bounds.upper, tRight) && tRight < tmax;

			// traverse closest first
			if (hitLeft && hitRight && (tLeft < tRight))
			{
				int tmp = node.leftIndex;
				node.leftIndex = node.rightIndex;
				node.rightIndex = tmp;
			}

			if (hitLeft)
				stack[count++] = node.leftIndex;

			if (hitRight)
				stack[count++] = node.rightIndex;			
		}
	}		

	// output results
	if (query.closestT < FLT_MAX)
	{
		t = query.closestT;
		u = query.closestU;
		v = query.closestV;
		w = query.closestW;	
		tri = query.closestTri;
		triNormal = query.closestNormal;

		return true;
	}
	else
	{
		return false;
	}
					
}



CUDA_CALLABLE bool inline IntersectRayMeshOld(const MeshGeometry& mesh, const Vec3& origin, const Vec3& dir, float tmax, float& t, float& u, float& v, float& w, int& tri, Vec3& triNormal)
{

	MeshQuery query(mesh, origin, dir);


	QueryRay(mesh.nodes, query, origin, dir);


	if (query.closestT < FLT_MAX)
	{
		t = query.closestT;
		u = query.closestU;
		v = query.closestV;
		w = query.closestW;	
		tri = query.closestTri;
		triNormal = query.closestNormal;

		return true;
	}
	else
	{
		return false;
	}
			

}

//-------------
// pdf that a point and dir were sampled by the light (assuming the ray hits the shape)

CUDA_CALLABLE inline float LightArea(const Primitive& p)
{
	switch (p.type)
	{
		case eSphere:
		{
			float area = 4.0f*kPi*p.sphere.radius*p.sphere.radius;  
			return area;
		}
		case ePlane:
		{
			return 0.0f;
		}
		case eMesh:
		{
			return p.mesh.area*p.endTransform.s;
		}
	};

	return 0.0f;
}

CUDA_CALLABLE inline void LightSample(const Primitive& p, float time, Vec3& pos, Vec3& normal, Random& rand)
{
	Transform transform = InterpolateTransform(p.startTransform, p.endTransform, time);

	switch (p.type)
	{
		case eSphere:
		{
			float u1, u2;
			Sample2D(rand, u1, u2);

			// todo: handle scaling in transform matrix
			pos = TransformPoint(transform, UniformSampleSphere(u1, u2)*p.sphere.radius);			
			normal = Normalize(pos-transform.p);
			return;
		}
		case ePlane:
		{
			assert(0);
			return;
		}
		case eMesh:
		{
			float r = rand.Randf();

			const float* triPtr = LowerBound(p.mesh.cdf, p.mesh.cdf+p.mesh.numIndices/3, r);
			int tri = Min(int(triPtr-p.mesh.cdf), p.mesh.numIndices/3 - 1);

			float u, v;
			UniformSampleTriangle(rand, u, v);
			
			// interpolate tri data
			int i0 = fetchInt(p.mesh.indices, tri*3+0);
			int i1 = fetchInt(p.mesh.indices, tri*3+1);
			int i2 = fetchInt(p.mesh.indices, tri*3+2);

			const Vec3 a = fetchVec3(p.mesh.positions, i0);
			const Vec3 b  = fetchVec3(p.mesh.positions, i1);
			const Vec3 c = fetchVec3(p.mesh.positions, i2);

			const Vec3 n1 = fetchVec3(p.mesh.normals, i0);
			const Vec3 n2  = fetchVec3(p.mesh.normals, i1);
			const Vec3 n3 = fetchVec3(p.mesh.normals, i2);

	        pos = TransformPoint(transform, u*a + v*b + (1.0f-u-v)*c);
	        normal = SafeNormalize(TransformVector(transform, u*n1 + v*n2 + (1.0f-u-v)*n3));
			return;
		}
	}
}

struct Ray
{
	CUDA_CALLABLE inline Ray(const Vec3& o, const Vec3& d, float t) : origin(o), dir(d), time(t) {}

	Vec3 origin;
	Vec3 dir;

	float time;
};

CUDA_CALLABLE inline bool Intersect(const Primitive& p, const Ray& ray, float& outT, Vec3* outNormal)
{
	Transform transform = InterpolateTransform(p.startTransform, p.endTransform, ray.time);

	switch (p.type)
	{
		case eSphere:
		{
			float minT, maxT;
			Vec3 n;

			bool hit = IntersectRaySphere(transform.p, p.sphere.radius*transform.s, ray.origin, ray.dir, minT, maxT, &n);

			if (hit)
			{
				outT = minT;
				*outNormal = n;
			}
			return hit;
		}
		case ePlane:
		{
			bool hit = IntersectRayPlane(ray.origin, ray.dir, (const Vec4&)p.plane, outT);
			
			if (hit && outNormal)
			{
				*outNormal = (const Vec3&)p.plane;
			}

			return hit;
		}
		case eMesh:
		{
			Vec3 localOrigin = InverseTransformPoint(transform, ray.origin);
			Vec3 localDir = InverseTransformVector(transform, ray.dir);

			float t, u, v, w;
			int tri;
			Vec3 triNormal;

			// transform ray to mesh space
			bool hit = IntersectRayMesh(p.mesh, localOrigin, localDir, FLT_MAX, t, u, v, w, tri, triNormal);
			
			if (hit)
			{
				// interpolate vertex normals
				int i0 = fetchInt(p.mesh.indices, tri*3+0);
				int i1 = fetchInt(p.mesh.indices, tri*3+1);
				int i2 = fetchInt(p.mesh.indices, tri*3+2);

				const Vec3 n1 = fetchVec3(p.mesh.normals, i0);
				const Vec3 n2  = fetchVec3(p.mesh.normals, i1);
				const Vec3 n3 = fetchVec3(p.mesh.normals, i2);

				Vec3 smoothNormal = u*n1 + v*n2 + w*n3;

				// ensure smooth normal lies on the same side of the geometric normal
				if (Dot(smoothNormal, triNormal) < 0.0f)
					smoothNormal *= -1.0f;

				outT = t;
				*outNormal = SafeNormalize(TransformVector(transform, smoothNormal), triNormal);
			}			

			return hit;
		}
	}

	return false;
}

