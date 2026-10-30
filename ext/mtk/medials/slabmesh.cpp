#include "medials/slabmesh.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <cassert>

SlabMesh::SlabMesh(const std::string &filename)
{
    if (!filename.empty())
    {
        load(filename);
    }
    nv = V.rows();
    ne = E.rows();
    nf = F.rows();
}

void SlabMesh::load(const std::string &filename)
{
    std::vector<vec4> vertices;
    std::vector<Eigen::Vector2i> edges;
    std::vector<Eigen::Vector3i> faces;

    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string prefix;
        ss >> prefix;

        if (prefix == "v")
        {
            scalar x, y, z, r;
            ss >> x >> y >> z >> r;
            vertices.emplace_back(x, y, z, r);
        }
        else if (prefix == "e")
        {
            int i, j;
            ss >> i >> j;
            edges.emplace_back(i, j);
        }
        else if (prefix == "f")
        {
            int i, j, k;
            ss >> i >> j >> k;
            faces.emplace_back(i, j, k);
        }
    }

    // Fill V and R
    int n = vertices.size();
    V.resize(n, 3);
    R.resize(n);
    for (int i = 0; i < n; ++i)
    {
        V.row(i) = vertices[i].head<3>();
        R(i) = vertices[i][3];
    }

    // Fill E0
    E0.resize(edges.size(), 2);
    for (size_t i = 0; i < edges.size(); ++i)
        E0.row(i) = edges[i];

    // Fill F
    F.resize(faces.size(), 3);
    for (size_t i = 0; i < faces.size(); ++i)
        F.row(i) = faces[i];

    // Construct full edge list: E0 + face-derived edges
    Eigen::MatrixXi all_edges;
    Eigen::MatrixXi EF;
    if (F.rows() > 0)
    {
        std::set<std::pair<int, int>> unique_edges;
        for (int face = 0; face < F.rows(); ++face)
        {
            for (int local = 0; local < 3; ++local)
            {
                int a = F(face, local);
                int b = F(face, (local + 1) % 3);
                unique_edges.emplace(std::min(a, b), std::max(a, b));
            }
        }
        EF.resize(static_cast<int>(unique_edges.size()), 2);
        int edge = 0;
        for (const auto &entry : unique_edges)
        {
            EF(edge, 0) = entry.first;
            EF(edge, 1) = entry.second;
            ++edge;
        }
        all_edges.resize(E0.rows() + EF.rows(), 2);
        all_edges << E0, EF;
    }
    else
    {
        all_edges.resize(E0.rows(), 2);
        all_edges = E0;
    }
    E = all_edges;
}

void SlabMesh::export_ma(const std::string &filename) const
{
    assert(filename.size() > 3 && filename.substr(filename.size() - 3) == ".ma");
    std::ofstream file(filename);
    file << V.rows() << " " << E0.rows() << " " << F.rows() << "\n";

    for (int i = 0; i < V.rows(); ++i)
        file << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << " " << R(i) << "\n";

    for (int i = 0; i < E0.rows(); ++i)
        file << "e " << E0(i, 0) << " " << E0(i, 1) << "\n";

    for (int i = 0; i < F.rows(); ++i)
        file << "f " << F(i, 0) << " " << F(i, 1) << " " << F(i, 2) << "\n";
}

void SlabMesh::export_ply(const std::string &filename) const
{
    assert(filename.size() > 4 && filename.substr(filename.size() - 4) == ".ply");
    std::ofstream file(filename);

    int n_faces = E.rows() + F.rows();
    file << "ply\nformat ascii 1.0\n";
    file << "element vertex " << V.rows() << "\n";
    file << "property float x\nproperty float y\nproperty float z\nproperty float r\n";
    file << "element face " << n_faces << "\n";
    file << "property list uchar int vertex_indices\nend_header\n";

    for (int i = 0; i < V.rows(); ++i)
        file << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << " " << R(i) << "\n";

    for (int i = 0; i < E.rows(); ++i)
        file << "2 " << E(i, 0) << " " << E(i, 1) << "\n";

    for (int i = 0; i < F.rows(); ++i)
        file << "3 " << F(i, 0) << " " << F(i, 1) << " " << F(i, 2) << "\n";
}
