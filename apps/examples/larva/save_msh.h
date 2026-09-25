#pragma once

#include <string>
#include <uipc/uipc.h>
#include <igl/writeMSH.h>

void save_msh(std::string_view file_name, const uipc::geometry::SimplicialComplex& sc)
{
    using namespace Eigen;
    using namespace uipc::geometry;

    // ---- V ----------------------------------------------------------------
    auto pos = sc.positions().view();
    MatrixXd V(pos.size(), 3);
    for(int i = 0; i < V.rows(); ++i)
        V.row(i) = pos[i];

    // ---- Tet --------------------------------------------------------
    auto topo = sc.tetrahedra().topo().view();
    MatrixXi Tet(topo.size(), 4);
    for(int i = 0; i < Tet.rows(); ++i)
        Tet.row(i) = topo[i];

    
    MatrixXi Tri, TriTag, TetTag;
    std::vector<std::string> XFields, EFields;
    std::vector<MatrixXd> XF, TriF, TetF;

    EFields.push_back("active_modulus");
    auto active_modulus = sc.tetrahedra().find<double>("active_modulus");
    auto active_modulus_view = active_modulus->view();

    VectorXd values(topo.size());
    for(int i = 0; i < values.rows(); ++i)
        values(i) = active_modulus_view[i];
    TetF.push_back(values);
    
    EFields.push_back("muscle_id");
    auto muscle_id = sc.tetrahedra().find<double>("muscle_id");
    auto muscle_id_view = muscle_id->view();

    for(int i = 0; i < values.rows(); ++i)
        values(i) = muscle_id_view[i];
    TetF.push_back(values);

    // seems like we have to add values to faces even if unused
    VectorXd padding;
    TriF.push_back(padding);
    TriF.push_back(padding);
    
    // ---- write ------------------------------------------------------------
    bool ok = igl::writeMSH(std::string{file_name}, V, Tri, Tet, TriTag, TetTag,
                            XFields, XF, EFields, TriF, TetF);
    UIPC_ASSERT(ok, "msh: failed to write `{}`.", file_name);
}