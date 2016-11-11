#include "mesh.h"
#include "maths.h"
#include "util.h"

#include <map>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

void Mesh::DuplicateVertex(int i)
{
	assert(positions.size() > i);	
	positions.push_back(positions[i]);
	
	if (normals.size() > i)
		normals.push_back(normals[i]);
}

void Mesh::Normalize(float s)
{
	Vec3 lower, upper;
	GetBounds(lower, upper);
	Vec3 edges = upper-lower;

	Transform(TranslationMatrix(Vec3(-lower)));

	float maxEdge = max(edges.x, max(edges.y, edges.z));
	Transform(ScaleMatrix(s/maxEdge));
}

void Mesh::CalculateNormals()
{
	normals.resize(0);
	normals.resize(positions.size());

	int numTris = indices.size()/3;

	for (int i=0; i < numTris; ++i)
	{
		int a = indices[i*3+0];
		int b = indices[i*3+1];
		int c = indices[i*3+2];
		
		Vec3 n = Cross(positions[b]-positions[a], positions[c]-positions[a]);

		normals[a] += n;
		normals[b] += n;
		normals[c] += n;
	}

	int numVertices = int(positions.size());

	for (int i=0; i < numVertices; ++i)
		normals[i] = ::SafeNormalize(normals[i]);
}

namespace 
{

    enum PlyFormat
    {
        eAscii,
        eBinaryBigEndian,
		eBinaryLittleEndian
    };

    template <typename T>
    T PlyRead(ifstream& s, PlyFormat format)
    {
        T data = eAscii;

        switch (format)
        {
            case eAscii:
            {
                s >> data;
                break;
            }
            case eBinaryBigEndian:
            {
                char c[sizeof(T)];
                s.read(c, sizeof(T));
                reverse(c, c+sizeof(T));
                data = *(T*)c;
                break;
            }      
			case eBinaryLittleEndian:
			{
			    char c[sizeof(T)];
                s.read(c, sizeof(T));
                data = *(T*)c;
                break;
			}
			default:
				assert(0);
        }

        return data;
    }

} // namespace anonymous

Mesh* ImportMesh(const char* path)
{
	const char* ext = strrchr(path, '.');

	Mesh* mesh = NULL;

	if (strcmp(ext, ".ply") == 0)
		mesh = ImportMeshFromPly(path);
	else if (strcmp(ext, ".obj") == 0)
		mesh = ImportMeshFromObj(path);
	else if (strcmp(ext, ".bin") == 0)
		mesh = ImportMeshFromBin(path);

	if (mesh && strcmp(ext, ".bin") != 0)
	{
		mesh->Normalize();
		mesh->CalculateNormals();
		mesh->RebuildBVH();
	}

	return mesh;
}

Mesh* ImportMeshFromPly(const char* path)
{
    ifstream file(path, ios_base::in | ios_base::binary);

    if (!file)
        return NULL;

    // some scratch memory
    const int kMaxLineLength = 1024;
    char buffer[kMaxLineLength];

    //double startTime = GetSeconds();

    file >> buffer;
    if (strcmp(buffer, "ply") != 0)
        return NULL;

    PlyFormat format = eAscii;

    int numFaces = 0;
    int numVertices = 0;

    const int kMaxProperties = 16;
    int numProperties = 0; 
    float properties[kMaxProperties];

    bool vertexElement = false;

    while (file)
    {
        file >> buffer;

        if (strcmp(buffer, "element") == 0)
        {
            file >> buffer;

            if (strcmp(buffer, "face") == 0)
            {                
                vertexElement = false;
                file >> numFaces;
            }

            else if (strcmp(buffer, "vertex") == 0)
            {
                vertexElement = true;
                file >> numVertices;
            }
        }
        else if (strcmp(buffer, "format") == 0)
        {
            file >> buffer;
            if (strcmp(buffer, "ascii") == 0)
            {
                format = eAscii;
            }
            else if (strcmp(buffer, "binary_big_endian") == 0)
            {
                format = eBinaryBigEndian;
            }
			else
			{
				format = eBinaryLittleEndian;
			}
        }
        else if (strcmp(buffer, "property") == 0)
        {
            if (vertexElement)
                ++numProperties;
        }
        else if (strcmp(buffer, "end_header") == 0)
        {
            break;
        }
    }

    // eat newline
    char nl;
    file.read(&nl, 1);
	
	// debug
	printf ("Loaded mesh: %s numFaces: %d numVertices: %d format: %d numProperties: %d\n", path, numFaces, numVertices, format, numProperties);

    Mesh* mesh = new Mesh;

    mesh->positions.resize(numVertices);
    mesh->normals.resize(numVertices);
    mesh->indices.reserve(numFaces*3);

    // read vertices
    for (int v=0; v < numVertices; ++v)
    {
        for (int i=0; i < numProperties; ++i)
        {
            properties[i] = PlyRead<float>(file, format);
        }

        mesh->positions[v] = Vec3(properties[0], properties[1], properties[2]);
        mesh->normals[v] = Vec3(0.0f, 0.0f, 0.0f);
    }

    // read indices
    for (int f=0; f < numFaces; ++f)
    {
        int numIndices = PlyRead<int>(file, format);
		int indices[4];

		for (int i=0; i < numIndices; ++i)
		{
			indices[i] = PlyRead<int>(file, format);
		}

		switch (numIndices)
		{
		case 3:
			mesh->indices.push_back(indices[0]);
			mesh->indices.push_back(indices[1]);
			mesh->indices.push_back(indices[2]);
			break;
		case 4:
			mesh->indices.push_back(indices[0]);
			mesh->indices.push_back(indices[1]);
			mesh->indices.push_back(indices[2]);

			mesh->indices.push_back(indices[2]);
			mesh->indices.push_back(indices[3]);
			mesh->indices.push_back(indices[0]);
			break;

		default:
			assert(!"invalid number of indices, only support tris and quads");
			break;
		};

		// calculate vertex normals as we go
        Vec3& v0 = mesh->positions[indices[0]];
        Vec3& v1 = mesh->positions[indices[1]];
        Vec3& v2 = mesh->positions[indices[2]];

        Vec3 n = SafeNormalize(Cross(v1-v0, v2-v0), Vec3(0.0f, 1.0f, 0.0f));

		for (int i=0; i < numIndices; ++i)
		{
	        mesh->normals[indices[i]] += n;
	    }
	}

    for (int i=0; i < numVertices; ++i)
    {
        mesh->normals[i] = SafeNormalize(mesh->normals[i], Vec3(0.0f, 1.0f, 0.0f));
    }

    //cout << "Imported mesh " << path << " in " << (GetSeconds()-startTime)*1000.f << "ms" << endl;

    return mesh;

}

// map of Material name to Material
struct VertexKey
{
	VertexKey() :  v(0), vt(0), vn(0) {}
	
	int v, vt, vn;
	
	bool operator == (const VertexKey& rhs) const
	{
		return v == rhs.v && vt == rhs.vt && vn == rhs.vn;
	}
	
	bool operator < (const VertexKey& rhs) const
	{
		if (v != rhs.v)
			return v < rhs.v;
		else if (vt != rhs.vt)
			return vt < rhs.vt;
		else
			return vn < rhs.vn;
	}
};

void Mesh::RebuildBVH()
{
	const int numTris = indices.size()/3;
	
	if (numTris)
	{
		std::vector<Bounds> triangleBounds(numTris);

		for (int i=0; i < numTris; ++i)
		{
			const Vec3 a = positions[indices[i*3+0]];
			const Vec3 b = positions[indices[i*3+1]];
			const Vec3 c = positions[indices[i*3+2]];

			triangleBounds[i].AddPoint(a);
			triangleBounds[i].AddPoint(b);
			triangleBounds[i].AddPoint(c);
		}
	
		BVHBuilder builder;
		bvh = builder.Build(&triangleBounds[0], numTris);
	}

    RebuildCDF();
}

void Mesh::RebuildCDF()
{
    int numTris = indices.size()/3;

    float totalArea = 0.0f;
    cdf.resize(0);

    for (int i=0; i < numTris; ++i)
    {
        const Vec3 a = positions[indices[i*3+0]];
        const Vec3 b = positions[indices[i*3+1]];
        const Vec3 c = positions[indices[i*3+2]];

        const float area = 0.5f*Length(Cross(b-a, c-a));

        totalArea += area;

        cdf.push_back(totalArea);
    }

    // normalize cdf
    for (int i=0; i < numTris; ++i)
    {
        cdf[i] /= totalArea;
    }

    // save total area
    area = totalArea;
}

Mesh* ImportMeshFromObj(const char* path)
{
    double start = GetSeconds();

    ifstream file(path);

    if (!file)
        return NULL;

    Mesh* m = new Mesh();

    vector<Vec3> positions;
    vector<Vec3> normals;
    vector<Vec2> texcoords;
    vector<Vec3> colors;
    vector<int>& indices = m->indices;

    //typedef unordered_map<VertexKey, int, MemoryHash<VertexKey> > VertexMap;
    typedef map<VertexKey, int> VertexMap;
    VertexMap vertexLookup;	

    // some scratch memory
    const int kMaxLineLength = 1024;
    char buffer[kMaxLineLength];

    //double startTime = GetSeconds();

    while (file)
    {
        file >> buffer;
	
        if (strcmp(buffer, "vn") == 0)
        {
            // normals
            float x, y, z;
            file >> x >> y >> z;

            normals.push_back(Vec3(x, y, z));
        }
        else if (strcmp(buffer, "vt") == 0)
        {
            // texture coords
            float u, v;
            file >> u >> v;

            texcoords.push_back(Vec2(u, v));
        }
        else if (buffer[0] == 'v')
        {
            // positions
            float x, y, z;
            file >> x >> y >> z;

            positions.push_back(Vec3(x, y, z));
        }
        else if (buffer[0] == 's' || buffer[0] == 'g' || buffer[0] == 'o')
        {
            // ignore smoothing groups, groups and objects
            char linebuf[256];
            file.getline(linebuf, 256);		
        }
        else if (strcmp(buffer, "mtllib") == 0)
        {
            // ignored
            std::string MaterialFile;
            file >> MaterialFile;
        }		
        else if (strcmp(buffer, "usemtl") == 0)
        {
            // read Material name
            std::string materialName;
            file >> materialName;
        }
        else if (buffer[0] == 'f')
        {
            // faces
            int faceIndices[4];
            int faceIndexCount = 0;

            for (int i=0; i < 4; ++i)
            {
                VertexKey key;

                file >> key.v;

				if (!file.eof())
				{
					// failed to read another index continue on
					if (file.fail())
					{
						file.clear();
						break;
					}

					if (file.peek() == '/')
					{
						file.ignore();

						if (file.peek() != '/')
						{
							file >> key.vt;
						}

						if (file.peek() == '/')
						{
							file.ignore();
							file >> key.vn;
						}
					}

					// convert relative coordinates to absolute
					if (key.v < 0)
						key.v = positions.size() + key.v + 1;
					
					if (key.vn < 0)
						key.vn = normals.size() + key.vn + 1;
					
					if (key.vt < 0)
						key.vt = texcoords.size() + key.vt + 1;

					// find / add vertex, index
					VertexMap::iterator iter = vertexLookup.find(key);

					if (iter != vertexLookup.end())
					{
						faceIndices[faceIndexCount++] = iter->second;
					}
					else
					{
						// add vertex
						int newIndex = m->positions.size();
						faceIndices[faceIndexCount++] = newIndex;

						vertexLookup.insert(make_pair(key, newIndex)); 	

						// push back vertex data
						assert(key.v > 0);

						m->positions.push_back(positions[key.v-1]);
						
						// normal [optional]
						if (key.vn)
						{
							m->normals.push_back(normals[key.vn-1]);
						}

						// texcoord [optional]
						if (key.vt)
						{
							//m->texcoords[0].push_back(texcoords[key.vt-1]);
						}
					}
				}
            }

            if (faceIndexCount == 3)
            {
                // a triangle
                indices.insert(indices.end(), faceIndices, faceIndices+3);
            }
            else if (faceIndexCount == 4)
            {
                // a quad, triangulate clockwise
                indices.insert(indices.end(), faceIndices, faceIndices+3);

                indices.push_back(faceIndices[2]);
                indices.push_back(faceIndices[3]);
                indices.push_back(faceIndices[0]);
            }
            else
            {
                cout << "Face with more than 4 vertices are not suppoted" << endl;
            }

        }		
        else if (buffer[0] == '#')
        {
            // comment
            char linebuf[256];
            file.getline(linebuf, 256);
        }
    }

    // calculate normals if none specified in file
    m->normals.resize(m->positions.size());

    const int numFaces = indices.size()/3;
    for (int i=0; i < numFaces; ++i)
    {
        int a = indices[i*3+0];
        int b = indices[i*3+1];
        int c = indices[i*3+2];

        Vec3& v0 = m->positions[a];
        Vec3& v1 = m->positions[b];
        Vec3& v2 = m->positions[c];

        Vec3 n = SafeNormalize(Cross(v1-v0, v2-v0), Vec3(0.0f, 1.0f, 0.0f));

        m->normals[a] += n;
        m->normals[b] += n;
        m->normals[c] += n;
    }

    for (int i=0; i < m->normals.size(); ++i)
    {
        m->normals[i] = SafeNormalize(m->normals[i], Vec3(0.0f, 1.0f, 0.0f));
    }
        
    double end = GetSeconds();

    //cout << "Imported mesh " << path << " in " << (GetSeconds()-startTime)*1000.f << "ms" << endl;
	m->RebuildBVH();

    double endBuild = GetSeconds();

    printf("Imported mesh %s (%d vertices, %d triangles) in %fms, build BVH in %fms\n", path, int(m->positions.size()), int(m->indices.size()/3), (end-start)*1000.0f, (endBuild-end)*1000.0f);



    return m;
}


Mesh* ImportMeshFromObjNew(const char* path)
{
    double start = GetSeconds();

    FILE* file = fopen(path, "r");

    if (!file)
    {
        printf("Could not open %s\n", path);
        return NULL;
    }

    Mesh* m = new Mesh();

    vector<Vec3> positions;
    vector<Vec3> normals; 
    vector<Vec2> texcoords;
    vector<Vec3> colors;
    vector<int>& indices = m->indices;

    //typedef unordered_map<VertexKey, int, MemoryHash<VertexKey> > VertexMap;
    typedef map<VertexKey, int> VertexMap;
    VertexMap vertexLookup; 

    // some scratch memory
    const int kMaxLineLength = 1024;
    char line[kMaxLineLength];

    //double startTime = GetSeconds();

    while (!feof(file))
    {
        fgets(line, kMaxLineLength, file);

        Vec3 v;
        if (line[0] == 'v')
        {
            if (line[1] == ' ')
            {
                sscanf(&line[2], "%f %f %f", &v.x, &v.y, &v.z);
                positions.push_back(v);
            }
            else if (line[1] == 'n')
            {
                sscanf(&line[2], "%f %f %f", &v.x, &v.y, &v.z);
                normals.push_back(v);
            }
            else if (line[1] == 't')
            {
                sscanf(&line[2], "%f %f", &v.x, &v.y);
                texcoords.push_back(Vec2(v.x, v.y));
            }
        }
        else if (line[0] == 's' || line[0] == 'g' || line[0] == 'o')
        {
            // ignore smoothing group  
        }
        /*
        else if (sscanf(line, " mtllib ") == 0)
        {
            // ignore material libs
            //std::string MaterialFile;
            //file >> MaterialFile;
        }       
        else if (sscanf(line, "usemtl") == 0)
        {
            // read Material name
            //std::string materialName;
            //file >> materialName;
        }
        */
        else if (line[0] == 'f')
        {
            // faces
            int faceIndices[4];
            int faceIndexCount = 0;
            int attribCount = 0;

            // eat "f "
            int cursor = 2;
            int start = cursor;

            VertexKey key;
            int* keyptr = (int*)(&key);

            for(;;)
            {
                if (line[cursor] == '/' || line[cursor] == ' ' || line[cursor] == 0)
                {
                    // read index
                    sscanf(&line[start], "%d", &keyptr[attribCount]);

                    attribCount++;

                    start = cursor+1;

                }

                // end of vertex definition, add to lookup
                if (line[cursor] == ' ' || line[cursor] == 0)
                {
                    attribCount = 0;

                    // find / add vertex, index
                    VertexMap::iterator iter = vertexLookup.find(key);

                    if (iter != vertexLookup.end())
                    {
                        faceIndices[faceIndexCount] = iter->second;
                    }
                    else
                    {
                        // add vertex
                        int newIndex = m->positions.size();
                        faceIndices[faceIndexCount] = newIndex;

                        vertexLookup.insert(make_pair(key, newIndex));  

                        // push back vertex data
                        assert(key.v > 0);

                        m->positions.push_back(positions[key.v-1]);
                        
                        // normal [optional]
                        if (key.vn)
                        {
                            m->normals.push_back(normals[key.vn-1]);
                        }

                        // texcoord [optional]
                        if (key.vt)
                        {
                           // m->texcoords[0].push_back(texcoords[key.vt-1]);
                        }
                    }
            
                    faceIndexCount++;
                }
    
                if (line[cursor] == 0)
                    break;
                
                ++cursor;
            }

           // printf("%d %d %d\n", faceIndices[0], faceIndices[1], faceIndices[2]);;

            if (faceIndexCount == 3)
            {
                // a triangle
                indices.insert(indices.end(), faceIndices, faceIndices+3);
            }
            else if (faceIndexCount == 4)
            {
                // a quad, triangulate clockwise
                indices.insert(indices.end(), faceIndices, faceIndices+3);

                indices.push_back(faceIndices[2]);
                indices.push_back(faceIndices[3]);
                indices.push_back(faceIndices[0]);
            }
            else
            {
                cout << "Face with more than 4 vertices are not supported" << endl;
            }

        }       
        /*
        else if (line[0] == '#')
        {
            // comment
        }
        */
    }

    // calculate normals if none specified in file
    m->normals.resize(m->positions.size());

    const int numFaces = indices.size()/3;
    for (int i=0; i < numFaces; ++i)
    {
        int a = indices[i*3+0];
        int b = indices[i*3+1];
        int c = indices[i*3+2];

        Vec3& v0 = m->positions[a];
        Vec3& v1 = m->positions[b];
        Vec3& v2 = m->positions[c];

        Vec3 n = SafeNormalize(Cross(v1-v0, v2-v0), Vec3(0.0f, 1.0f, 0.0f));

        m->normals[a] += n;
        m->normals[b] += n;
        m->normals[c] += n;
    }

    for (int i=0; i < m->normals.size(); ++i)
    {
        m->normals[i] = SafeNormalize(m->normals[i], Vec3(0.0f, 1.0f, 0.0f));
    }
        
    double end = GetSeconds();

    //cout << "Imported mesh " << path << " in " << (GetSeconds()-startTime)*1000.f << "ms" << endl;
    m->RebuildBVH();

    double endBuild = GetSeconds();

    printf("Imported mesh %s (%d vertices, %d triangles) in %fms, build BVH in %fms\n", path, int(m->positions.size()), int(m->indices.size()/3), (end-start)*1000.0f, (endBuild-end)*1000.0f);

    fclose(file);

    return m;
}

Mesh* ImportMeshFromBin(const char* path)
{
	double start = GetSeconds();

	FILE* f = fopen(path, "rb");

	if (f)
	{
		int numVertices;
		int numIndices;
		int numNodes;

		fread(&numVertices, sizeof(numVertices), 1, f);
		fread(&numIndices, sizeof(numIndices), 1, f);
		fread(&numNodes, sizeof(numNodes), 1, f);

		Mesh* m = new Mesh();
		m->positions.resize(numVertices);
		m->normals.resize(numVertices);
		m->indices.resize(numIndices);
		m->cdf.resize(numIndices/3);
		
		m->bvh.nodes = new BVHNode[numNodes];
		m->bvh.numNodes = numNodes;

		fread(&m->positions[0], sizeof(Vec3)*numVertices, 1, f);
		fread(&m->normals[0], sizeof(Vec3)*numVertices, 1, f);
		fread(&m->indices[0], sizeof(int)*numIndices, 1, f);
		fread(m->bvh.nodes, sizeof(BVHNode)*numNodes, 1, f);
		
		fread(&m->area, sizeof(float), 1, f);
		fread(&m->cdf[0], sizeof(float)*numIndices/3, 1, f);

		fclose(f);

		double end = GetSeconds();

		printf("Imported mesh %s in %f ms\n", path, (end-start)*1000.0f);

		return m;
	}

	return NULL;
}

void ExportMeshToBin(const char* path, const Mesh* m)
{
	FILE* f = fopen(path, "wb");

	if (f)
	{
		int numVertices = m->positions.size();
		int numIndices = m->indices.size();
		int numNodes = m->bvh.numNodes;

		fwrite(&numVertices, sizeof(numVertices), 1, f);
		fwrite(&numIndices, sizeof(numIndices), 1, f);
		fwrite(&numNodes, sizeof(numNodes), 1, f);

		// write data blocks
		fwrite(&m->positions[0], sizeof(Vec3)*numVertices, 1, f);
		fwrite(&m->normals[0], sizeof(Vec3)*numVertices, 1, f);		
		fwrite(&m->indices[0], sizeof(int)*numIndices, 1, f);
		fwrite(m->bvh.nodes, sizeof(BVHNode)*numNodes, 1, f);

		// cdf for light sampling
		fwrite(&m->area, sizeof(float), 1, f);
		fwrite(&m->cdf[0], sizeof(float)*numIndices/3, 1, f);

		fclose(f);
	}
}


void ExportMeshToObj(const char* path, const Mesh& m)
{
	ofstream file(path);

    if (!file)
        return;

	file << "# positions" << endl;

	for (int i=0; i < m.positions.size(); ++i)
	{
		Vec3 v = m.positions[i];
		file << "v " << v.x << " " << v.y << " " << v.z << endl;
	}
/*
	file << "# texcoords" << endl;

	for (int i=0; i < m.texcoords[0].size(); ++i)
	{
		Vec2 t = m.texcoords[0][i];
		file << "vt " << t.x << " " << t.y << endl;
	}
*/
	file << "# normals" << endl;

	for (int i=0; i < m.normals.size(); ++i)
	{
		Vec3 n = m.normals[0][i];
		file << "vn " << n.x << " " << n.y << " " << n.z << endl;
	}

	file << "# faces" << endl;

	for (int i=0; i < m.indices.size()/3; ++i)
	{
		int j = i+1;

		// no sharing, assumes there is a unique position, texcoord and normal for each vertex
		file << "f " << j << "/"/* << j << */"/" << j << endl;
	}
}

void Mesh::AddMesh(Mesh& m)
{
    int offset = positions.size();

    // add new vertices
    positions.insert(positions.end(), m.positions.begin(), m.positions.end());
    normals.insert(normals.end(), m.normals.begin(), m.normals.end());

    // add new indices with offset
    for (int i=0; i < m.indices.size(); ++i)
    {
        indices.push_back(m.indices[i]+offset);
    }    
}


void Mesh::Transform(const Mat44& m)
{
    for (int i=0; i < positions.size(); ++i)
    {
        positions[i] = TransformPoint(m, positions[i]);
        normals[i] = TransformVector(m, normals[i]);
    }
}

void Mesh::GetBounds(Vec3& outMinExtents, Vec3& outMaxExtents) const
{
    Vec3 minExtents(REAL_MAX);
    Vec3 maxExtents(-REAL_MAX);

    // calculate face bounds
    for (int i=0; i < positions.size(); ++i)
    {
        const Vec3& a = positions[i];

        minExtents = Min(a, minExtents);
        maxExtents = Max(a, maxExtents);
    }

    outMinExtents = Vec3(minExtents);
    outMaxExtents = Vec3(maxExtents);
}

Mesh* CreateQuadMesh(float size, float y)
{
    int indices[] = { 0, 1, 2, 2, 3, 0 };
    Vec3 positions[4];
    Vec3 normals[4];

    positions[0] = Vec3(-size, y, size);
    positions[1] = Vec3(size, y, size);
    positions[2] = Vec3(size, y, -size);
    positions[3] = Vec3(-size, y, -size);
    
    normals[0] = Vec3(0.0f, 1.0f, 0.0f);
    normals[1] = Vec3(0.0f, 1.0f, 0.0f);
    normals[2] = Vec3(0.0f, 1.0f, 0.0f);
    normals[3] = Vec3(0.0f, 1.0f, 0.0f);

    Mesh* m = new Mesh();
    m->indices.insert(m->indices.begin(), indices, indices+6);
    m->positions.insert(m->positions.begin(), positions, positions+4);
    m->normals.insert(m->normals.begin(), normals, normals+4);

    return m;
}

Mesh* CreateDiscMesh(float radius, int segments)
{
	const int numVerts = 1 + segments;

	Mesh* m = new Mesh();
	m->positions.resize(numVerts);
	m->normals.resize(numVerts);

	m->positions[0] = Vec3(0.0f);
	m->positions[1] = Vec3(0.0f, 0.0f, radius);

	for (int i=1; i <= segments; ++i)
	{
		int nextVert = (i+1)%numVerts;

		if (nextVert == 0)
			nextVert = 1;
		
		m->positions[nextVert] = Vec3(radius*sin((float(i)/segments)*k2Pi), 0.0f, radius*cos((float(i)/segments)*k2Pi));
		m->normals[nextVert] = Vec3(0.0f, 1.0f, 0.0f);

		m->indices.push_back(0);
		m->indices.push_back(i);
		m->indices.push_back(nextVert);		
	}
	
	return m;
}

Mesh* CreateTetrahedron()
{
	Mesh* m = new Mesh();

	const float dimValue = 1.0f / sqrtf(2.0f);
	const Vec3 vertices[4] =
	{
		Vec3(-1.0f, 0.0f, -dimValue),
		Vec3(1.0f, 0.0f, -dimValue),
		Vec3(0.0f, 0.0f + 1.0f, dimValue),
		Vec3(0.0f, 0.0f, dimValue)
	};

	const int indices[12] = 
	{
		//winding order is counter-clockwise
		0, 2, 1,
		2, 3, 1,
		2, 0, 3,
		3, 0, 1
	};

	m->positions.assign(vertices, vertices+4);
	m->indices.assign(indices, indices+12);

	m->CalculateNormals();

	return m;
	
}





Mesh* CreateSphere(int slices, int segments, float radius)
{
	float dTheta = kPi / slices;
	float dPhi = k2Pi / segments;

	int vertsPerRow = segments + 1;

	Mesh* mesh = new Mesh();

	for (int i = 0; i <= slices; ++i)
	{
		float theta = dTheta*i;

		for (int j = 0; j <= segments; ++j)
		{
			float phi = dPhi*j;

			float x = sinf(theta)*cosf(phi);
			float y = cosf(theta);
			float z = sinf(theta)*sinf(phi);

			mesh->positions.push_back(Vec3(x, y, z)*radius);
			mesh->normals.push_back(Vec3(x, y, z));

			if (i > 0 && j > 0)
			{
				int a = i*vertsPerRow + j;
				int b = (i - 1)*vertsPerRow + j;
				int c = (i - 1)*vertsPerRow + j - 1;
				int d = i*vertsPerRow + j - 1;

				// add a quad for this slice
				mesh->indices.push_back(b);
				mesh->indices.push_back(a);
				mesh->indices.push_back(d);

				mesh->indices.push_back(b);
				mesh->indices.push_back(d);
				mesh->indices.push_back(c);
			}
		}
	}

	return mesh;
}

Mesh* CreateCapsule(int slices, int segments, float radius, float halfHeight)
{
	float dTheta = kPi / (slices * 2);
	float dPhi = k2Pi / segments;

	int vertsPerRow = segments + 1;

	Mesh* mesh = new Mesh();

	float theta = 0.0f;

	for (int i = 0; i <= 2 * slices + 1; ++i)
	{
		for (int j = 0; j <= segments; ++j)
		{
			float phi = dPhi*j;

			float x = sinf(theta)*cosf(phi);
			float y = cosf(theta);
			float z = sinf(theta)*sinf(phi);

			// add y offset based on which hemisphere we're in
			float yoffset = (i < slices) ? halfHeight : -halfHeight;

			mesh->positions.push_back(Vec3(x, y, z)*radius + Vec3(0.0f, yoffset, 0.0f));
			mesh->normals.push_back(Vec3(x, y, z));

			if (i > 0 && j > 0)
			{
				int a = i*vertsPerRow + j;
				int b = (i - 1)*vertsPerRow + j;
				int c = (i - 1)*vertsPerRow + j - 1;
				int d = i*vertsPerRow + j - 1;

				// add a quad for this slice
				mesh->indices.push_back(b);
				mesh->indices.push_back(a);
				mesh->indices.push_back(d);

				mesh->indices.push_back(b);
				mesh->indices.push_back(d);
				mesh->indices.push_back(c);
			}
		}

		// don't update theta for the middle slice
		if (i != slices)
			theta += dTheta;
	}

	return mesh;
}
