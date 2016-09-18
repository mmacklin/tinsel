#pragma once

#include <math.h>

#include <assert.h>
#include <algorithm>
#include <limits>

#include <stdint.h>

#define USE_DOUBLE_PRECISION 0

#if USE_DOUBLE_PRECISION
typedef double Real;
#else
typedef float Real;
#endif

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif


// short hand for max values
#define REAL_MAX std::numeric_limits<Real>::max()

// this header defines C++ types and helpers for manipulating basic vector types
// matrices are stored in column major order, with column vectors (OpenGL style)

const Real kPi = 3.141592653589793;
const Real k2Pi = 3.141592653589793*2.0;
const Real kInvPi = 1.0/kPi;
const Real kInv2Pi = 1.0/k2Pi;

inline Real DegToRad(Real t) { return t * (kPi/180.0); }
inline Real RadToDeg(Real t) { return t * (180.0/kPi); }

inline Real Sqr(Real x) { return x*x; }
inline Real Cube(Real x) { return x*x*x; }

inline Real Sign(Real x) { return x < 0.0f ? -1.0f : 1.0f; }

inline Real Sqrt(Real x) { return sqrt(x); }

template <typename T>
inline void Swap(T& a, T& b)
{
	T temp = a;
	a = b;
	b = temp;
}

template <typename T>
inline T Min(T a, T b) { return std::min(a, b); }

template <typename T>
inline T Max(T a, T b) { return std::max(a, b); }

template <typename T>
inline T Clamp(T x, T lower, T upper)
{
	return Min(Max(x, lower), upper);
}

template <typename T>
inline T Abs(T x)
{
	if (x < 0.0)
		return -x;
	else
		return x;
}

template <typename T>
inline T Lerp(T a, T b, Real t)
{
	return a + (b-a)*t;
}

// generic size matrix multiply, result must not alias a or b
template <int m, int n, int p>
inline void MatrixMultiply(Real* result, const Real* a, const Real* b)
{
	for (int i=0; i < m; ++i)
	{
		for (int j=0; j < p; ++j)
		{
			Real t = 0.0f;
			for (int k=0; k < n; ++k)
			{
				t += a[i+k*m]*b[k+j*n];
			}

			result[i+j*m] = t;
		}
	}
}

// generic size matrix transpose, result must not alias a
template <int m, int n>
inline void MatrixTranspose(Real* result, const Real* a)
{
	for (int i=0; i < m; ++i)
	{
		for (int j=0; j < n; ++j)
		{
			result[j+i*n] = a[i+j*m];
		}
	}
}

template <int m, int n>
inline void MatrixAdd(Real* result, const Real* a, const Real* b)
{
	for (int j=0; j < n; ++j)
	{
		for (int i=0; i < m; ++i)
		{
			const int idx = j*m+i; 
			result[idx] = a[idx] + b[idx];
		}
	}
}


template <int m, int n>
inline void MatrixSub(Real* result, const Real* a, const Real* b)
{
	for (int j=0; j < n; ++j)
	{
		for (int i=0; i < m; ++i)
		{
			const int idx = j*m+i; 
			result[idx] = a[idx] - b[idx];
		}
	}
}

template <int m, int n>
inline void MatrixScale(Real* result, const Real* a, const Real s)
{
	for (int j=0; j < n; ++j)
	{
		for (int i=0; i < m; ++i)
		{
			const int idx = j*m+i; 
			result[idx] = a[idx]*s;
		}
	}
}


//--------------------

struct Vec2
{
	Vec2() : x(0.0f), y(0.0f) {}
	Vec2(Real x) : x(x), y(x) {}
	Vec2(Real x, Real y) : x(x), y(y) {}

	Real operator[](int index) const { assert(index < 2); return (&x)[index]; }
	Real& operator[](int index) { assert(index < 2); return (&x)[index]; }

	Real x;
	Real y;
};

inline Vec2 Max(const Vec2& a, const Vec2& b)
{
	return Vec2(std::max(a.x, b.x), std::max(a.y, b.y));
}

inline Vec2 Min(const Vec2& a, const Vec2& b)
{
	return Vec2(std::min(a.x, b.x), std::min(a.y, b.y));
}

inline Vec2 operator-(const Vec2& a) { return Vec2(-a.x, -a.y); }
inline Vec2 operator+(const Vec2& a, const Vec2& b) { return Vec2(a.x+b.x, a.y+b.y); }
inline Vec2 operator-(const Vec2& a, const Vec2& b) { return Vec2(a.x-b.x, a.y-b.y); }
inline Vec2 operator*(const Vec2& a, Real s) { return Vec2(a.x*s, a.y*s); }
inline Vec2 operator*(Real s, const Vec2& a) { return a*s; }
inline Vec2 operator*(const Vec2& a, const Vec2& b) { return Vec2(a.x*b.x, a.y*b.y); }
inline Vec2 operator/(const Vec2& a, Real s) { return a*(1.0/s); }
inline Vec2 operator/(const Vec2& a, const Vec2& b) { return Vec2(a.x/b.x, a.y/b.y); }

inline Vec2& operator+=(Vec2& a, const Vec2& b) { return a = a+b; }
inline Vec2& operator-=(Vec2& a, const Vec2& b) { return a = a-b; }
inline Vec2& operator*=(Vec2& a, Real s) { a.x *= s; a.y *= s; return a; }
inline Vec2& operator*=(Vec2& a, const Vec2& b) { a.x *= b.x; a.y *= b.y; return a; }
inline Vec2& operator/=(Vec2& a, Real s) { Real rcp=1.0/s; a.x *= rcp; a.y *= rcp; return a; }
inline Vec2& operator/=(Vec2& a, const Vec2& b) { a.x /= b.x; a.y /= b.y; return a; }

inline Vec2 PerpCCW(const Vec2& v) { return Vec2(-v.y, v.x); }
inline Vec2 PerpCW(const Vec2& v) { return Vec2( v.y, -v.x); }
inline Real Dot(const Vec2& a, const Vec2& b) { return a.x*b.x + a.y*b.y; }
inline Real LengthSq(const Vec2& a) { return Dot(a,a); }
inline Real Length(const Vec2& a) { return sqrt(LengthSq(a)); }
inline Vec2 Normalize(const Vec2& a) { return a/Length(a); }
inline Vec2 SafeNormalize(const Vec2& a, const Vec2& fallback=Vec2(0.0)) 
{
	Real l=Length(a); 
	if (l > 0.0)
		return a/l;
	else
		return fallback;
}

//--------------------
struct Vec4;

struct Vec3
{
	Vec3() : x(0.0), y(0.0), z(0.0) {}
	Vec3(Real x) : x(x), y(x), z(x) {}
	Vec3(Real x, Real y, Real z) : x(x), y(y), z(z) {}
	Vec3(const Vec2& v, Real z) : x(v.x), y(v.y), z(z) {}
	explicit Vec3(const Vec4& v);

	Real operator[](int index) const { assert(index < 3); return (&x)[index]; }
	Real& operator[](int index) { assert(index < 3); return (&x)[index]; }

	Real x;
	Real y;
	Real z;
};

inline Vec3 operator-(const Vec3& a) { return Vec3(-a.x, -a.y, -a.z); }
inline Vec3 operator+(const Vec3& a, const Vec3& b) { return Vec3(a.x+b.x, a.y+b.y, a.z+b.z); }
inline Vec3 operator-(const Vec3& a, const Vec3& b) { return Vec3(a.x-b.x, a.y-b.y, a.z-b.z); }
inline Vec3 operator*(const Vec3& a, Real s) { return Vec3(a.x*s, a.y*s, a.z*s); }
inline Vec3 operator*(Real s, const Vec3& a) { return a*s; }
inline Vec3 operator*(const Vec3& a, const Vec3& b) { return Vec3(a.x*b.x, a.y*b.y, a.z*b.z); }
inline Vec3 operator/(const Vec3& a, Real s) { return a*(1.0/s); }
inline Vec3 operator/(const Vec3& a, const Vec3& b) { return Vec3(a.x/b.x, a.y/b.y, a.z/b.z); }

inline Vec3& operator+=(Vec3& a, const Vec3& b) { return a = a+b; }
inline Vec3& operator-=(Vec3& a, const Vec3& b) { return a = a-b; }
inline Vec3& operator*=(Vec3& a, Real s) { a.x *= s; a.y *= s;  a.z *= s; return a; }
inline Vec3& operator/=(Vec3& a, Real s) { Real rcp=1.0/s; a.x *= rcp; a.y *= rcp; a.z *= rcp; return a; }

inline Vec3 Cross(const Vec3& a, const Vec3& b) { return Vec3(a.y*b.z - b.y*a.z, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); }
inline Real Dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline Real LengthSq(const Vec3& a) { return Dot(a,a); }
inline Real Length(const Vec3& a) { return sqrt(LengthSq(a)); }
inline Vec3 Normalize(const Vec3& a) { return a/Length(a); }
inline Vec3 SafeNormalize(const Vec3& a, const Vec3& fallback=Vec3(0.0))
{
	Real m = LengthSq(a);
	
	if (m > 0.0)
	{
		return a * (1.0/sqrt(m));
	}
	else
	{
		return fallback;
	}
}

inline Vec3 Abs(const Vec3& a) { return Vec3(Abs(a.x), Abs(a.y), Abs(a.z)); }

inline Vec3 Max(const Vec3& a, const Vec3& b)
{
	return Vec3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.y));
}

inline Vec3 Min(const Vec3& a, const Vec3& b)
{
	return Vec3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}


//--------------------

struct Vec4
{
	Vec4() : x(0.0), y(0.0), z(0.0), w(0.0) {}
	Vec4(Real x) : x(x), y(x), z(x), w(0.0) {}
	Vec4(Real x, Real y, Real z, Real w=0.0f) : x(x), y(y), z(z), w(w) {}
	Vec4(const Vec3& v, Real w) : x(v.x), y(v.y), z(v.z), w(w) {}

	Real operator[](int index) const { assert(index < 4); return (&x)[index]; }
	Real& operator[](int index) { assert(index < 4); return (&x)[index]; }

	Real x;
	Real y;
	Real z;
	Real w;
};

inline Vec4 operator-(const Vec4& a) { return Vec4(-a.x, -a.y, -a.z, -a.w); }
inline Vec4 operator+(const Vec4& a, const Vec4& b) { return Vec4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }
inline Vec4 operator-(const Vec4& a, const Vec4& b) { return Vec4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w); }
inline Vec4 operator*(const Vec4& a, const Vec4& b) { return Vec4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w); }
inline Vec4 operator*(const Vec4& a, Real s) { return Vec4(a.x*s, a.y*s, a.z*s, a.w*s); }
inline Vec4 operator*(Real s, const Vec4& a) { return a*s; }
inline Vec4 operator/(const Vec4& a, Real s) { return a*(1.0/s); }

inline Vec4& operator+=(Vec4& a, const Vec4& b) { return a = a+b; }
inline Vec4& operator-=(Vec4& a, const Vec4& b) { return a = a-b; }
inline Vec4& operator*=(Vec4& a, const Vec4& s) { a.x *= s.x; a.y *= s.y;  a.z *= s.z; a.w *= s.w; return a; }
inline Vec4& operator*=(Vec4& a, Real s) { a.x *= s; a.y *= s;  a.z *= s; a.w *= s; return a; }
inline Vec4& operator/=(Vec4& a, Real s) { Real rcp=1.0/s; a.x *= rcp; a.y *= rcp; a.z *= rcp; a.w *= rcp; return a; }

inline Vec4 Cross(const Vec4& a, const Vec4& b);
inline Real Dot(const Vec4& a, const Vec4& b) { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }
inline Real LengthSq(const Vec4& a) { return Dot(a,a); }
inline Real Length(const Vec4& a) { return sqrt(LengthSq(a)); }
inline Vec4 Normalize(const Vec4& a) { return a/Length(a); }
inline Vec4 SafeNormalize(const Vec4& a, const Vec4& fallback=Vec4(0.0));

inline Vec3::Vec3(const Vec4& v) : x(v.x), y(v.y), z(v.z) {}

// matrix22
struct Mat22
{

	Mat22()
	{
		m[0][0] = 0.0;
		m[0][1] = 0.0;
		m[1][0] = 0.0;
		m[1][1] = 0.0;		
	}

	Mat22(Real m11, Real m12, Real m21, Real m22)
	{
		m[0][0] = m11;
		m[0][1] = m21;
		m[1][0] = m12;
		m[1][1] = m22;
	}

	Mat22(const Vec2& c1, const Vec2& c2)
	{
		SetCol(0, c1);
		SetCol(1, c2);
	}

	static Mat22 Identity() 
	{
		return Mat22(Vec2(1.0, 0.0), Vec2(0.0, 1.0));
	}

	Vec2 GetCol(int index) const { return Vec2(m[index][0], m[index][1]); }
	void SetCol(int index, const Vec2& v)
	{ 
		m[index][0] = v.x;
		m[index][1] = v.y;
	}

	Real m[2][2];
};

inline Vec2 Multiply(const Mat22& a, const Vec2& v)
{
	Vec2 result;
	MatrixMultiply<2, 2, 1>(&result.x, &a.m[0][0], &v.x);
	return result;
}

inline Mat22 operator+(const Mat22& a, const Mat22& b)
{
	Mat22 s;
	MatrixAdd<2, 2>(&s.m[0][0], &a.m[0][0], &b.m[0][0]);
	return s;
}

inline Mat22 operator-(const Mat22& a, const Mat22& b)
{
	Mat22 s;
	MatrixSub<2, 2>(&s.m[0][0], &a.m[0][0], &b.m[0][0]);
	return s;
}

inline Mat22 operator*(const Mat22& a, const Mat22& b)
{
	Mat22 result;
	MatrixMultiply<2, 2, 2>(&result.m[0][0], &a.m[0][0], &b.m[0][0]);
	return result;
}

// matrix multiplcation
inline Vec2 operator*(const Mat22& a, const Vec2& v) { return Multiply(a, v); }

// scalar multiplication
inline Mat22 operator*(const Mat22& a, Real s) 
{
	Mat22 result;
	MatrixScale<2, 2>(&result.m[0][0], &a.m[0][0], s); 
	return result;
}

inline Mat22 operator*(Real s, const Mat22& a) { return a*s; }

// unary negation
inline Mat22 operator-(const Mat22& a) { return -1.0f*a; }

// generate a counter clockwise rotation by theta radians
inline Mat22 RotationMatrix(Real theta)
{
	Real cosTheta = cos(theta);
	Real sinTheta = sin(theta);

	Real m[2][2] = 
	{
		{  cosTheta, sinTheta },
		{ -sinTheta, cosTheta }
	};

	return *(Mat22*)m;
}

// derivative of a rotation matrix w.r.t. theta
inline Mat22 RotationMatrixDerivative(Real theta)
{
	Real cosTheta = cos(theta);
	Real sinTheta = sin(theta);

	Real m[2][2] = 
	{
		{  -sinTheta, cosTheta },
		{  -cosTheta, -sinTheta }
	};

	return *(Mat22*)m;
}

inline Real Determinant(const Mat22& a)
{
	return a.m[0][0]*a.m[1][1]-a.m[1][0]*a.m[0][1];	
}

inline Real Trace(const Mat22& a) 
{
	return a.m[0][0] + a.m[1][1]; 
}

inline Mat22 Outer(const Vec2& a, const Vec2& b)
{
	return Mat22(a*b.x, a*b.y);
}

inline Mat22 Inverse(const Mat22& a, Real* det)
{
	*det = Determinant(a);
	if (*det != 0.0f)
	{
		return (1.0/(*det))*Mat22(a.m[1][1], -a.m[1][0], -a.m[0][1], a.m[0][0]);
	}
	else
		return a;
}

inline Mat22 Transpose(const Mat22& a)
{
	Mat22 t;
	MatrixTranspose<2,2>(&t.m[0][0], &a.m[0][0]);
	return t;
}

//---------------------------

struct Quat
{
	Quat() : x(0.0), y(0.0), z(0.0), w(1.0) {}
	Quat(Real x, Real y, Real z, Real w) : x(x), y(y), z(z), w(w) {}
	Quat(Vec3 axis, Real angle)
	{
		const Real s = sin(angle*0.5);
		const Real c = cos(angle*0.5);

		x = axis.x*s;
		y = axis.y*s;
		z = axis.z*s;
		w = c;
	}

	Vec3 GetImaginary() { return Vec3(x, y, z); }

	Real x;
	Real y;
	Real z;
	Real w;	// real part	
};

inline Quat operator-(const Quat& a) { return Quat(-a.x, -a.y, -a.z, -a.w); }
inline Quat operator+(const Quat& a, const Quat& b) { return Quat(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }
inline Quat operator-(const Quat& a, const Quat& b) { return Quat(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w); }
inline Quat operator*(const Quat& a, Real s) { return Quat(a.x*s, a.y*s, a.z*s, a.w*s); }
inline Quat operator*(const Quat& a, const Quat& b) 
{
	return Quat(a.w*b.x + b.w*a.x + a.y*b.z - b.y*a.z,
				  a.w*b.y + b.w*a.y + a.z*b.x - b.z*a.x,
				  a.w*b.z + b.w*a.z + a.x*b.y - b.x*a.y,
				  a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z);
}
inline Quat operator*(Real s, const Quat& a) { return a*s; }
inline Quat operator/(const Quat& a, Real s) { return a*(1.0/s); }

inline Quat& operator+=(Quat& a, const Quat& b) { return a = a+b; }
inline Quat& operator-=(Quat& a, const Quat& b) { return a = a-b; }
inline Quat& operator*=(Quat& a, const Quat& b) { return a = a*b; }
inline Quat& operator*=(Quat& a, Real s) { a.x *= s; a.y *= s;  a.z *= s; return a; }
inline Quat& operator/=(Quat& a, Real s) { Real rcp=1.0/s; a.x *= rcp; a.y *= rcp; a.z *= rcp; return a; }

inline Quat Normalize(const Quat& q)
{
	Real length = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;
	Real rcpLength = 1.0/length;

	return q*rcpLength;
}

inline Quat Conjugate(const Quat& q) { return Quat(-q.x, -q.y, -q.z, q.w); }


inline Vec3 Rotate(const Quat& q, const Vec3& v)
{
	// q'*v*q
	Quat t = Conjugate(q)*Quat(v.x, v.y, v.z, 0.0)*q;
	return t.GetImaginary();
}

inline Vec3 operator*(const Quat& q, const Vec3& v)
{
	return Rotate(q, v);
}



// ---------------------
// represents a rigid body transformation

struct Transform
{
	// transform
	Transform() : p(0.0) {}
	Transform(const Vec3& v, const Quat& r=Quat()) : p(v), r(r) {}

	Transform operator*(const Transform& rhs) const
	{
		return Transform(Rotate(r, rhs.p) + p, r*rhs.r);
	}

	Vec3 p;
	Quat r;
};

inline Transform Inverse(const Transform& transform)
{
	Transform t;
	t.r = Conjugate(transform.r);
	t.p = -Rotate(t.r, transform.p);	

	return t;
}

inline Vec3 TransformVector(const Transform& t, const Vec3& v)
{
	return t.r*v;
}

inline Vec3 TransformPoint(const Transform& t, const Vec3& v)
{
	return t.r*v + t.p;
}

inline Vec3 InverseTransformVector(const Transform& t, const Vec3& v)
{
	return Conjugate(t.r)*v;
}

inline Vec3 InverseTransformPoint(const Transform& t, const Vec3& v)
{
	return Conjugate(t.r)*(v-t.p);
}

// ----------------------

struct Mat33
{
	Mat33() { memset(this, 0, sizeof(*this)); }

	Mat33(Real m11, Real m12, Real m13, 
			Real m21, Real m22, Real m23,
			Real m31, Real m32, Real m33)
	{
		// col 1
		m[0][0] = m11;
		m[0][1] = m21;
		m[0][2] = m31;
		
		// col 2
		m[1][0] = m12;
		m[1][1] = m22;
		m[1][2] = m32;

		// col 3
		m[2][0] = m13;
		m[2][1] = m23;
		m[2][2] = m33;
	}

	Mat33(const Vec3& c1, const Vec3& c2, const Vec3& c3)
	{
		SetCol(0, c1);
		SetCol(1, c2);
		SetCol(2, c3);
	}

	Mat33(const Quat& q)
	{
		*this = Mat33(1.0-2.0*(q.y*q.y-q.z*q.z), 2.0*(q.x*q.y+q.w*q.z), 2.0*(q.x*q.z-q.w*q.y),
						2.0*(q.x*q.y-q.w*q.z), 1.0-2.0*(q.x*q.x-q.z*q.z), 2.0*(q.y*q.z-q.w*q.x),
						2.0*(q.x*q.z+q.w*q.y), 2.0*(q.y*q.z-q.w*q.x), 1.0-2.0*(q.x*q.x-q.y*q.y));
	}

	Vec3 GetCol(int index) const { return Vec3(m[index][0], m[index][1], m[index][2]); }
	void SetCol(int index, const Vec3& v)
	{
		m[index][0] = v.x;
		m[index][1] = v.y;
		m[index][2] = v.z;
	}

	Real m[3][3];	
};

inline Mat33 Outer(const Vec3& a, const Vec3& b)
{
	return Mat33(a*b.x, a*b.y, a*b.z);
}

inline Vec3 Multiply(const Mat33& a, const Vec3& v)
{
	Vec3 result;
	MatrixMultiply<3, 3, 1>(&result.x, &a.m[0][0], &v.x);
	return result;
}

inline Mat33 Multiply(const Mat33& a, const Mat33& b)
{
	Mat33 result;
	MatrixMultiply<3, 3, 3>(&result.m[0][0], &a.m[0][0], &b.m[0][0]);
	return result;
}

inline Vec3 operator*(const Mat33& a, const Vec3& v) { return Multiply(a, v); }
inline Mat33 operator*(const Real s, const Mat33& a) 
{
	Mat33 result;
	MatrixScale<3,3>(&result.m[0][0], &a.m[0][0], s);
	return result;
}

inline Mat33 operator*(const Mat33& a, const Real s) { return s*a; }
inline Mat33 operator*(const Mat33& a, const Mat33& b) { return Multiply(a, b); }

inline Mat33 operator-(const Mat33& a, const Mat33& b) 
{
	Mat33 result; 
	MatrixSub<3, 3>(&result.m[0][0], &a.m[0][0], &b.m[0][0]);
	return result;
}

inline Mat33 operator+(const Mat33& a, const Mat33& b)
{
	Mat33 result; 
	MatrixAdd<3, 3>(&result.m[0][0], &a.m[0][0], &b.m[0][0]);
	return result;
}

inline Mat33& operator+=(Mat33& a, const Mat33& b) { return a = a+b; }
inline Mat33& operator-=(Mat33& a, const Mat33& b) { return a = a-b; }

inline Vec2 TransformVector(const Mat33& a, const Vec2& v)
{
	Vec2 result;
	result.x = a.m[0][0]*v.x + a.m[1][0]*v.y;
	result.y = a.m[0][1]*v.x + a.m[1][1]*v.y;
	return result;
}

inline Vec2 TransformPoint(const Mat33& a, const Vec2& v)
{
	Vec2 result;
	result.x = a.m[0][0]*v.x + a.m[1][0]*v.y + a.m[2][0];
	result.y = a.m[0][1]*v.x + a.m[1][1]*v.y + a.m[2][1];
	return result;
}

// returns the skew-symmetric matrix that performs cross(v, x) when multiplied on the left
inline Mat33 Skew(const Vec3& v)
{
	return Mat33( 0.0f, -v.z, v.y,
					v.z, 0.0f, -v.x,
				   -v.y, v.x, 0.0f);
}

//-------------------

struct Mat44
{
	Mat44() { memset(this, 0, sizeof(*this)); }

	Mat44(Real m11, Real m12, Real m13, Real m14, 
			Real m21, Real m22, Real m23, Real m24,
			Real m31, Real m32, Real m33, Real m34,
			Real m41, Real m42, Real m43, Real m44)
	{
		// col 1
		cols[0][0] = m11;
		cols[0][1] = m21;
		cols[0][2] = m31;
		cols[0][3] = m41;
		
		// col 2
		cols[1][0] = m12;
		cols[1][1] = m22;
		cols[1][2] = m32;
		cols[1][3] = m42;

		// col 3
		cols[2][0] = m13;
		cols[2][1] = m23;
		cols[2][2] = m33;
		cols[2][3] = m43;

		// col 4
		cols[3][0] = m14;
		cols[3][1] = m24;
		cols[3][2] = m34;
		cols[3][3] = m44;		

	}

	Mat44(const Vec4& c1, const Vec4& c2, const Vec4& c3, const Vec4& c4)
	{
		SetCol(0, c1);
		SetCol(1, c2);
		SetCol(2, c3);
		SetCol(3, c4);
	}


	Mat44(const Transform& t)
	{
		Mat33 r(t.r);

		SetCol(0, Vec4(r.GetCol(0), 0.0));
		SetCol(1, Vec4(r.GetCol(1), 0.0));
		SetCol(2, Vec4(r.GetCol(2), 0.0));
		SetCol(4, Vec4(t.p, 1.0));
	}

	static Mat44 Identity() 
	{
		return Mat44(1.0f, 0.0f, 0.0f, 0.0f,
					 0.0f, 1.0f, 0.0f, 0.0f,
					 0.0f, 0.0f, 1.0f, 0.0f,
					 0.0f, 0.0f, 0.0f, 1.0f);
	}	

	Vec4 GetCol(int index) const { return Vec4(cols[index][0], cols[index][1], cols[index][2], cols[index][3]); }
	void SetCol(int index, const Vec4& v)
	{
		cols[index][0] = v.x;
		cols[index][1] = v.y;
		cols[index][2] = v.z;
		cols[index][3] = v.w;
	}

	Real cols[4][4];	// column major format
};


inline Vec4 Multiply(const Mat44& a, const Vec4& v)
{
	Vec4 result;
	MatrixMultiply<4, 4, 1>(&result.x, &a.cols[0][0], &v.x);
	return result;
}

inline Mat44 Multiply(const Mat44& a, const Mat44& b)
{
	Mat44 result;
	MatrixMultiply<4, 4, 4>(&result.cols[0][0], &a.cols[0][0], &b.cols[0][0]);
	return result;
}

inline Vec4 operator*(const Mat44& a, const Vec4& v) { return Multiply(a, v); }
inline Mat44 operator*(const Real s, const Mat44& a) 
{
	Mat44 result;
	MatrixScale<4,4>(&result.cols[0][0], &a.cols[0][0], s);
	return result;
}

inline Mat44 operator*(const Mat44& a, const Real s) { return s*a; }
inline Mat44 operator*(const Mat44& a, const Mat44& b) { return Multiply(a, b); }

inline Mat44 operator-(const Mat44& a, const Mat44& b) 
{
	Mat44 result; 
	MatrixSub<4, 4>(&result.cols[0][0], &a.cols[0][0], &b.cols[0][0]);
	return result;
}

inline Mat44 operator+(const Mat44& a, const Mat44& b)
{
	Mat44 result; 
	MatrixAdd<4, 4>(&result.cols[0][0], &a.cols[0][0], &b.cols[0][0]);
	return result;
}

inline Mat44& operator+=(Mat44& a, const Mat44& b) { return a = a+b; }
inline Mat44& operator-=(Mat44& a, const Mat44& b) { return a = a-b; }

inline Vec3 TransformVector(const Mat44& a, const Vec3& v)
{
	Vec3 result;
	result.x = a.cols[0][0]*v.x + a.cols[1][0]*v.y + a.cols[2][0]*v.z;
	result.y = a.cols[0][1]*v.x + a.cols[1][1]*v.y + a.cols[2][1]*v.z;
	result.z = a.cols[0][2]*v.x + a.cols[1][2]*v.y + a.cols[2][2]*v.z;
	return result;
}

inline Vec3 TransformPoint(const Mat44& a, const Vec3& v)
{
	Vec3 result;
	result.x = a.cols[0][0]*v.x + a.cols[1][0]*v.y + a.cols[2][0]*v.z + a.cols[3][0];
	result.y = a.cols[0][1]*v.x + a.cols[1][1]*v.y + a.cols[2][1]*v.z + a.cols[3][1];
	result.z = a.cols[0][2]*v.x + a.cols[1][2]*v.y + a.cols[2][2]*v.z + a.cols[3][2];
	return result;
}

inline Mat44 Transpose(const Mat44& a)
{
	Mat44 t;
	MatrixTranspose<4,4>(&t.cols[0][0], &a.cols[0][0]);
	return t;
}

//-----------------

struct Bounds
{
	Bounds() : lower( REAL_MAX)
	           , upper(-REAL_MAX) {}

	Bounds(const Vec3& lower, const Vec3& upper) : lower(lower), upper(upper) {}

	Vec3 GetCenter() const { return 0.5*(lower+upper); }
	Vec3 GetEdges() const { return upper-lower; }

	void Expand(Real r)
	{
		lower -= Vec3(r);
		upper += Vec3(r);
	}

	bool Empty() const { return lower.x >= upper.x || lower.y >= upper.y; }

	void AddPoint(const Vec3& p)
	{
		lower = Min(lower, p);
		upper = Max(upper, p);
	}

	bool Overlaps(const Vec3& p) const
	{
		if (p.x < lower.x ||
			p.y < lower.y ||
			p.x > upper.x ||
			p.y > upper.y)
		{
			return false;
		}
		else
		{
			return true;
		}
	}

	bool Overlaps(const Bounds& b) const
	{
		if (lower.x > b.upper.x ||
			lower.y > b.upper.y ||
			upper.x < b.lower.x ||
			upper.y < b.lower.y)
		{
			return false;
		}
		else
		{
			return true;
		}
	}

	Vec3 lower;
	Vec3 upper;
};

inline Bounds TransformBounds(const Transform& xform, const Bounds& bounds)
{
	// take the sum of the +/- abs value along each cartesian axis
	Mat33 m = Mat33(xform.r);

	Vec3 halfEdgeWidth = bounds.GetEdges()*0.5;

	Vec3 x = Abs(m.GetCol(0))*halfEdgeWidth.x;
	Vec3 y = Abs(m.GetCol(1))*halfEdgeWidth.y;
	Vec3 z = Abs(m.GetCol(2))*halfEdgeWidth.z;

	Vec3 center = bounds.GetCenter();

	Vec3 lower = center + x + y + z;
	Vec3 upper = center - x - y - z;

	return Bounds(lower, upper);
}

inline Bounds Union(const Bounds& a, const Bounds& b) 
{
	return Bounds(Min(a.lower, b.lower), Max(a.upper, b.upper));
}

inline Bounds Intersection(const Bounds& a, const Bounds& b)
{
	return Bounds(Max(a.lower, b.lower), Min(a.upper, b.upper));
}

// -----------------
// random numbers

class Random
{
public:

	Random()
	{
		seed1 = 315645664;
		seed2 = seed1 ^ 0x13ab45fe;
	}

	inline unsigned int Rand()
	{
		seed1 = ( seed2 ^ ( ( seed1 << 5 ) | ( seed1 >> 27 ) ) ) ^ ( seed1*seed2 );
		seed2 = seed1 ^ ( ( seed2 << 12 ) | ( seed2 >> 20 ) );

		return seed1;
	}

	// returns a random number in the range [min, max)
	inline unsigned int Rand(unsigned int min, unsigned int max)
	{
		return min + Rand()%(max-min);
	}

	// returns random number between 0-1
	inline float Randf()
	{
		unsigned int value = Rand();
		unsigned int limit = 0xffffffff;

		return ( float )value*(1.0/( float )limit );
	}

	// returns random number between min and max
	inline float Randf(float min, float max)
	{
		//	return Lerp(min, max, ParticleRandf());
		float t = Randf();
		return (1.0-t)*min + t*(max);
	}

	// returns random number between 0-max
	inline float Randf(float max)
	{
		return Randf()*max;
	}

	unsigned int seed1;
	unsigned int seed2;
};

//----------------------
// bitwise operations

inline int Part1By1(int n)
{
	n=(n ^ (n << 8))&0x00ff00ff;
	n=(n ^ (n << 4))&0x0f0f0f0f;
	n=(n ^ (n << 2))&0x33333333;
	n=(n ^ (n << 1))&0x55555555;

	return n; 
}

// Takes values in the range [0,1] and assigns an index based on 16bit Morton codes
inline int Morton2(Real x, Real y)
{
	int ux = Clamp(int(x*1024), 0, 1023);
	int uy = Clamp(int(y*1024), 0, 1023);

	return (Part1By1(uy) << 1) + Part1By1(ux);
}

inline int Part1by2(int n)
{
	n = (n ^ (n << 16)) & 0xff0000ff;
	n = (n ^ (n <<  8)) & 0x0300f00f;
	n = (n ^ (n <<  4)) & 0x030c30c3;
	n = (n ^ (n <<  2)) & 0x09249249;

	return n;
}

// Takes values in the range [0, 1] and assigns an index based on 10bit Morton codes
inline int Morton3(Real x, Real y, Real z)
{
	int ux = Clamp(int(x*1024), 0, 1023);
	int uy = Clamp(int(y*1024), 0, 1023);
	int uz = Clamp(int(z*1024), 0, 1023);

	return (Part1by2(uz) << 2) | (Part1by2(uy) << 1) | Part1by2(ux);
}

// count number of leading zeros in a 32bit word
inline int CLZ(int x)
{
	int n;
	if (x == 0) return 32;
	for (n = 0; ((x & 0x80000000) == 0); n++, x <<= 1);
	return n;
}

//----------------
// geometric tests

inline void ProjectPointToLine(Vec2 p, Vec2 a, Vec2 b, Real& t)
{

}

inline Vec2 ClosestPointToLineSegment(Vec2 p, Vec2 a, Vec2 b, Real& t)
{
	Vec2 edge = b-a;
	Real edgeLengthSq = LengthSq(edge);
	
	if (edgeLengthSq == 0.0)
	{
		// degenerate edge, return first vertex
		t = 0.0;
		return a;
	}

	Vec2 delta = p-a;
	t = Dot(delta, edge)/edgeLengthSq;

	if (t <= 0.0)
	{
		return a;
	}
	else if (t >= 1.0)
	{
		return b;
	}
	else
	{
		return a + t*(b-a);
	}
}

// generates a transform matrix with v as the z axis, taken from PBRT
inline void BasisFromVector(const Vec3& w, Vec3* u, Vec3* v)
{
	if (fabs(w.x) > fabs(w.y))
	{
		Real invLen = 1.0 / sqrt(w.x*w.x + w.z*w.z);
		*u = Vec3(-w.z*invLen, 0.0f, w.x*invLen);
	}
	else
	{
		Real invLen = 1.0 / sqrt(w.y*w.y + w.z*w.z);
		*u = Vec3(0.0f, w.z*invLen, -w.y*invLen);
	}

	*v = Cross(w, *u);	
}


inline Vec3 UniformSampleSphere(Random& rand)
{
	float u1 = rand.Randf(0.0f, 1.0f);
	float u2 = rand.Randf(0.0f, 1.0f);

	float z = 1.f - 2.f * u1;
	float r = sqrtf(Max(0.f, 1.f - z*z));
	float phi = 2.f * kPi * u2;
	float x = r * cosf(phi);
	float y = r * sinf(phi);

	return Vec3(x, y, z);
}

inline Vec3 UniformSampleHemisphere(Random& rand)
{
	// generate a random z value
	float z = rand.Randf(0.0f, 1.0f);
	float w = sqrt(1.0f-z*z);

	float phi = k2Pi*rand.Randf(0.0f, 1.0f);
	float x = cos(phi)*w;
	float y = sin(phi)*w;

	return Vec3(x, y, z);
}

inline Vec2 UniformSampleDisc(Random& rand)
{
	float r = sqrt(rand.Randf(0.0f, 1.0f));
	float theta = k2Pi*rand.Randf(0.0f, 1.0f);

	return Vec2(r * cos(theta), r * sin(theta));
}

inline void UniformSampleTriangle(Random& rand, float& u, float& v)
{
	float r = sqrt(rand.Randf());
	u = 1.0f - r;
	v = rand.Randf() * r;
}

inline Vec3 CosineSampleHemisphere(Random& rand)
{
	Vec2 s = UniformSampleDisc(rand);
	float z = sqrt(Max(0.0f, 1.0f - s.x*s.x - s.y*s.y));

	return Vec3(s.x, s.y, z);
}

inline Vec3 SphericalToXYZ(float theta, float phi)
{
	float cosTheta = cos(theta);
	float sinTheta = sin(theta);

	return Vec3(sin(phi)*sinTheta, cosTheta, cos(phi)*sinTheta);
}

inline Mat44 AffineInverse(const Mat44& m)
{
	Mat44 inv;
	
	// transpose upper 3x3
	for (int c=0; c < 3; ++c)
	{
		for (int r=0; r < 3; ++r)
		{
			inv.cols[c][r] = m.cols[r][c];
		}
	}
	
	// multiply -translation by upper 3x3 transpose
	inv.cols[3][0] = -Dot(m.GetCol(3), m.GetCol(0));
	inv.cols[3][1] = -Dot(m.GetCol(3), m.GetCol(1));
	inv.cols[3][2] = -Dot(m.GetCol(3), m.GetCol(2));
	inv.cols[3][3] = 1.0f;

	return inv;	
}

inline Mat44 LookAtMatrix(const Vec3& viewer, const Vec3& target)
{
	// create a basis from viewer to target (OpenGL convention looking down -z)
	Vec3 forward = -Normalize(target-viewer);
	Vec3 up(0.0f, 1.0f, 0.0f);
	Vec3 left = Cross(up, forward);
	up = Cross(forward, left);

	Mat44 xform(left.x, up.x, forward.x, viewer.x,
	  			left.y, up.y, forward.y, viewer.y,
				left.z, up.z, forward.z, viewer.z,
				0.0f, 0.0f, 0.0f, 1.0f);

	return AffineInverse(xform);
}


// generate a rotation matrix around an axis, from PBRT p74
inline Mat44 RotationMatrix(float angle, const Vec3& axis)
{
	Vec3 a = Normalize(axis);
	float s = sinf(angle);
	float c = cosf(angle);

	float m[4][4];

	m[0][0] = a.x * a.x + (1.0f - a.x * a.x) * c;
	m[0][1] = a.x * a.y * (1.0f - c) + a.z * s;
	m[0][2] = a.x * a.z * (1.0f - c) - a.y * s;
	m[0][3] = 0.0f;

	m[1][0] = a.x * a.y * (1.0f - c) - a.z * s;
	m[1][1] = a.y * a.y + (1.0f - a.y * a.y) * c;
	m[1][2] = a.y * a.z * (1.0f - c) + a.x * s;
	m[1][3] = 0.0f;

	m[2][0] = a.x * a.z * (1.0f - c) + a.y * s;
	m[2][1] = a.y * a.z * (1.0f - c) - a.x * s;
	m[2][2] = a.z * a.z + (1.0f - a.z * a.z) * c;
	m[2][3] = 0.0f;

	m[3][0] = 0.0f;
	m[3][1] = 0.0f;
	m[3][2] = 0.0f;
	m[3][3] = 1.0f;

	return (Mat44&)m;
}

inline Mat44 TranslationMatrix(const Vec3& t)
{
	Mat44 m = Mat44::Identity();
	m.SetCol(3, Vec4(t, 1.0f));
	return m;
}


inline Mat44 ScaleMatrix(const Vec3& s)
{
	float m[4][4] = { {s.x, 0.0f, 0.0f, 0.0f },
					  { 0.0f, s.y, 0.0f, 0.0f},
					  { 0.0f, 0.0f, s.z, 0.0f},
					  { 0.0f, 0.0f, 0.0f, 1.0f} };

	return (Mat44&)m;
}

inline Mat44 OrthographicMatrix(float left, float right, float bottom, float top, float n, float f)
{
	
	float m[4][4] = { { 2.0f/(right-left), 0.0f, 0.0f, 0.0f },
					  { 0.0f, 2.0f/(top-bottom), 0.0f, 0.0f },			
					  { 0.0f, 0.0f, -2.0f/(f-n), 0.0f },
					  { -(right+left)/(right-left), -(top+bottom)/(top-bottom), -(f+n)/(f-n), 1.0f } };
	

	return (Mat44&)m;
}

// this is designed as a drop in replacement for gluPerspective
inline Mat44 ProjectionMatrix(float fov, float aspect, float znear, float zfar) 
{
	float f = 1.0f / tanf(DegToRad(fov*0.5f));
	float zd = znear-zfar;

	float view[4][4] = { { f/aspect, 0.0f, 0.0f, 0.0f },
						 { 0.0f, f, 0.0f, 0.0f },
						 { 0.0f, 0.0f, (zfar+znear)/zd, -1.0f },
						 { 0.0f, 0.0f, (2.0f*znear*zfar)/zd, 0.0f } };
 
	return (Mat44&)view;
}

typedef Vec4 Color;



inline Color YxyToXYZ(float Y, float x, float y)
{
	float X = x * (Y / y);
	float Z = (1.0f - x - y) * Y / y;

	return Color(X, Y, Z, 1.0f);
}

inline Color HSVToRGB( float h, float s, float v )
{
	float r, g, b;

	int i;
	float f, p, q, t;
	if( s == 0 ) {
		// achromatic (grey)
		r = g = b = v;
	}
	else
	{
		h *= 6.0f;			// sector 0 to 5
		i = int(floor( h ));
		f = h - i;			// factorial part of h
		p = v * ( 1 - s );
		q = v * ( 1 - s * f );
		t = v * ( 1 - s * ( 1 - f ) );
		switch( i ) {
			case 0:
				r = v;
				g = t;
				b = p;
				break;
			case 1:
				r = q;
				g = v;
				b = p;
				break;
			case 2:
				r = p;
				g = v;
				b = t;
				break;
			case 3:
				r = p;
				g = q;
				b = v;
				break;
			case 4:
				r = t;
				g = p;
				b = v;
				break;
			default:		// case 5:
				r = v;
				g = p;
				b = q;
				break;
		};
	}

	return Color(r, g, b);
}

inline Color XYZToLinear(float x, float y, float z)
{
	float c[4];
	c[0] =  3.240479f * x + -1.537150f * y + -0.498535f * z;
	c[1] = -0.969256f * x +  1.875991f * y +  0.041556f * z;
	c[2] =  0.055648f * x + -0.204043f * y +  1.057311f * z;
	c[3] = 1.0f;

	return Color(c[0], c[1], c[2], c[3]);
}

inline int ColorToRGBA8(const Color& c)
{
	union SmallColor
	{
		uint8_t u8[4];
		int u32;
	};

	SmallColor s;
	s.u8[0] = (uint8_t)(Clamp(c.x, 0.0f, 1.0f) * 255);
	s.u8[1] = (uint8_t)(Clamp(c.y, 0.0f, 1.0f) * 255);
	s.u8[2] = (uint8_t)(Clamp(c.z, 0.0f, 1.0f) * 255);
	s.u8[3] = (uint8_t)(Clamp(c.w, 0.0f, 1.0f) * 255);

	return s.u32;
}

inline Color LinearToSrgb(const Color& c)
{
	const float kInvGamma = 1.0f/2.2f;
	return Color(powf(c.x, kInvGamma), powf(c.y, kInvGamma), powf(c.z, kInvGamma), c.w); 
}

inline Color SrgbToLinear(const Color& c)
{
	const float kInvGamma = 2.2f;
	return Color(powf(c.x, kInvGamma), powf(c.y, kInvGamma), powf(c.z, kInvGamma), c.w); 
}




