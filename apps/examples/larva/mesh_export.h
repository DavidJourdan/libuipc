#pragma once

#include <string>
#include <uipc/uipc.h>
#include <igl/writeMSH.h>
#include <fstream>
#include <uipc/common/enumerate.h>

enum class save_type
{
    surface,
    msh,
};

using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;

void save_msh(std::string_view file_name, const uipc::geometry::SimplicialComplex& sc)
{
    using namespace Eigen;
    using namespace uipc::geometry;

    // ---- V ----------------------------------------------------------------
    auto     pos = sc.positions().view();
    MatrixXd V(pos.size(), 3);
    for(int i = 0; i < V.rows(); ++i)
        V.row(i) = pos[i];

    // ---- Tet --------------------------------------------------------
    auto     topo = sc.tetrahedra().topo().view();
    MatrixXi Tet(topo.size(), 4);
    for(int i = 0; i < Tet.rows(); ++i)
        Tet.row(i) = topo[i];

    MatrixXi                 Tri, TriTag, TetTag;
    std::vector<std::string> XFields, EFields;
    std::vector<MatrixXd>    XF, TriF, TetF;

    EFields.push_back("active_modulus");
    auto active_modulus      = sc.tetrahedra().find<double>("active_modulus");
    auto active_modulus_view = active_modulus->view();

    VectorXd values(topo.size());
    for(int i = 0; i < values.rows(); ++i)
        values(i) = active_modulus_view[i];
    TetF.push_back(values);

    EFields.push_back("muscle_id");
    auto muscle_id      = sc.tetrahedra().find<double>("muscle_id");
    auto muscle_id_view = muscle_id->view();

    for(int i = 0; i < values.rows(); ++i)
        values(i) = muscle_id_view[i];
    TetF.push_back(values);

    // seems like we have to add values to faces even if unused
    VectorXd padding;
    TriF.push_back(padding);
    TriF.push_back(padding);

    // ---- write ------------------------------------------------------------
    bool ok = igl::writeMSH(
        std::string{file_name}, V, Tri, Tet, TriTag, TetTag, XFields, XF, EFields, TriF, TetF);
    UIPC_ASSERT(ok, "msh: failed to write `{}`.", file_name);
}

void save_npy(std::string file_name, const std::span<Vector3>& V)
{
    std::ofstream file(file_name);
    UIPC_ASSERT(file.is_open(), "npz: failed to open `{}`.", file_name);

    // Magic string
    file.write("\x93NUMPY", 6);

    // Version 1.0
    file.put(0x01);
    file.put(0x00);

    // Create header dict
    std::ostringstream header;
    header << "{'descr': '<f8', 'fortran_order': False, 'shape': (";
    header << V.size() << ", " << 3 << "), }";

    // Pad header to multiple of 64 bytes (including length prefix)
    std::string header_str = header.str();
    size_t total_header_len = 6 + 2 + 2 + header_str.size();  // magic + version + len + dict
    size_t padding = (64 - (total_header_len % 64)) % 64;
    header_str.append(padding, ' ');
    header_str.push_back('\n');

    // Write header length (little-endian uint16)
    uint16_t header_len = header_str.size();
    file.write(reinterpret_cast<const char*>(&header_len), 2);

    // Write header
    file.write(header_str.c_str(), header_len);

    // Write data in row-major (C) order
    for(int i = 0; i < V.size(); ++i)
        file.write(reinterpret_cast<const char*>(V[i].data()), 3 * sizeof(double));

    file.close();
}

class MeshExporter
{
  public:
    MeshExporter(const ObjectGeometrySlots<SimplicialComplex>& geometry_slot, save_type type)
        : _type(type)
        , _geometry_slot(geometry_slot)
    {
        if(type == save_type::surface)
        {
            // larva outer surface
            auto larva_mesh = _geometry_slot.geometry->geometry();
            label_surface(larva_mesh);
            label_triangle_orient(larva_mesh);

            _larva_surface =
                std::make_shared<SimplicialComplex>(extract_surface(larva_mesh));

            // muscle surfaces
            auto V = larva_mesh.vertices().find<Vector3>(builtin::position)->view();
            auto tet_view = larva_mesh.tetrahedra().topo().view();
            auto muscle_id_view =
                larva_mesh.tetrahedra().find<double>("muscle_id")->view();

            vector<Vector4i> T;
            vector<int> muscle_ids;
            for(int i = 0; i < tet_view.size(); ++i)
            {
                if(muscle_id_view[i] > 0)
                {
                    T.push_back(tet_view[i]);
                    muscle_ids.push_back(static_cast<int>(muscle_id_view[i]));
                }
            }

            _muscle_mesh = std::make_shared<SimplicialComplex>(tetmesh(V, T));
            label_surface(*_muscle_mesh);
            label_triangle_orient(*_muscle_mesh);

            // transfer muscle_id from tets to triangles
            auto parent_id_view = _muscle_mesh->triangles().find<IndexT>("parent_id")->view();
            auto muscle_id_surface_view = view(*_muscle_mesh->triangles().create<int>("muscle_id"));
            for(int i = 0; i < parent_id_view.size(); ++i)
                muscle_id_surface_view[i] = muscle_ids[parent_id_view[i]];

            auto muscle_id_vertex_view = view(*_muscle_mesh->vertices().create<int>("muscle_id"));
            auto triangles = _muscle_mesh->triangles().topo().view();
            for(int i = 0; i < parent_id_view.size(); ++i)
            {
                muscle_id_vertex_view[triangles[i](0)] = muscle_ids[parent_id_view[i]];
                muscle_id_vertex_view[triangles[i](1)] = muscle_ids[parent_id_view[i]];
                muscle_id_vertex_view[triangles[i](2)] = muscle_ids[parent_id_view[i]];
            }

            _muscle_surface =
                std::make_shared<SimplicialComplex>(extract_surface(*_muscle_mesh));
        }
    }


    ~MeshExporter() = default;

    void save_scene(int frame_id)
    {
        if(_type == save_type::surface)
            save_surface(frame_id);
        else if(_type == save_type::msh)
            save_msh(fmt::format("{}scene_mesh{}.msh",
                                 AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE),
                                 frame_id),
                     _geometry_slot.geometry->geometry());
    }

    void save_surface(int frame_id)
    {
        SimplicialComplex larva_mesh = _geometry_slot.geometry->geometry();

        auto V_in = larva_mesh.positions().view();
        auto larva_out =
            view(*_larva_surface->vertices().find<Vector3>(builtin::position));
        auto muscle_out =
            view(*_muscle_surface->vertices().find<Vector3>(builtin::position));

        auto larva_is_surface_view =
            larva_mesh.vertices().find<IndexT>(builtin::is_surf)->view();
        auto muscle_is_surface_view =
            _muscle_mesh->vertices().find<IndexT>(builtin::is_surf)->view();

        IndexT larva_v_count  = 0;
        IndexT muscle_v_count = 0;
        for(int i = 0; i < larva_is_surface_view.size(); ++i)
        {
            if(larva_is_surface_view[i])
                larva_out[larva_v_count++] = V_in[i];

            if(muscle_is_surface_view[i])
                muscle_out[muscle_v_count++] = V_in[i];
        }

        // TODO write vertex positions in binary, no face indices (topology is redundant)
        SimplicialComplexIO io;
        auto output_path = AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE);
        io.write_obj(fmt::format("{}larva_surface{}.obj", output_path, frame_id),
                     *_larva_surface);
        io.write_obj(fmt::format("{}muscle_surface{}.obj", output_path, frame_id),
                     *_muscle_surface);
    }


    void save_vertices_binary(int frame_id)
    {
        SimplicialComplex larva_mesh = _geometry_slot.geometry->geometry();

        auto V_in = larva_mesh.positions().view();

        auto larva_is_surface_view =
            larva_mesh.vertices().find<IndexT>(builtin::is_surf)->view();
        auto muscle_is_surface_view =
            _muscle_mesh->vertices().find<IndexT>(builtin::is_surf)->view();

        std::vector<Vector3> larva_out(_larva_surface->vertices().size());
        std::vector<Vector3> muscle_out(_muscle_surface->vertices().size());

        IndexT larva_v_count  = 0;
        IndexT muscle_v_count = 0;
        for(int i = 0; i < larva_is_surface_view.size(); ++i)
        {
            if(larva_is_surface_view[i])
                larva_out[larva_v_count++] = V_in[i];

            if(muscle_is_surface_view[i])
                muscle_out[muscle_v_count++] = V_in[i];
        }

        auto output_path = AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE);

        save_npy(fmt::format("{}larva_vertices{}.npy", output_path, frame_id), larva_out);
        save_npy(fmt::format("{}muscle_vertices{}.npy", output_path, frame_id), muscle_out);
    }

    void muscle_groups_to_json(const std::string &filename)
    {
        auto labels_attr = _muscle_surface->vertices().find<int>("muscle_id");

        int max_muscle_id = 0;
        for(int muscle_id : labels_attr->view())
            if(muscle_id > max_muscle_id)
                max_muscle_id = muscle_id;

        std::vector<std::vector<int>> muscle_groups(max_muscle_id + 1);

        for(auto&& [i, muscle_id] : enumerate(labels_attr->view()))
        {
            if(muscle_id > 0)
                muscle_groups[muscle_id].push_back(i);
        }

        Json muscle_group_data;
        for(auto&& [i, muscle_group] : enumerate(muscle_groups))
        {
            if(muscle_group.size() > 0)
                muscle_group_data.emplace(std::to_string(i), muscle_group);
        }
        
        std::ofstream file(filename);
        file << muscle_group_data.dump();
    }

  private:
    save_type                              _type;
    ObjectGeometrySlots<SimplicialComplex> _geometry_slot;

    std::shared_ptr<SimplicialComplex> _larva_surface;
    std::shared_ptr<SimplicialComplex> _muscle_surface;
    std::shared_ptr<SimplicialComplex> _larva_mesh;
    std::shared_ptr<SimplicialComplex> _muscle_mesh;
};
