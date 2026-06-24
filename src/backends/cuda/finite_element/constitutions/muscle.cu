#include <finite_element/constitutions/muscle.h>
#include <finite_element/constitutions/muscle_function.h>
#include <finite_element/fem_3d_extra_constitution.h>
#include <finite_element/fem_exporter.h>
#include <finite_element/fem_utils.h>
#include <Eigen/Dense>
#include <utils/make_spd.h>
#include <utils/matrix_assembler.h>
// #include <uipc/core/muscle_controller.h>

namespace uipc::backend::cuda
{
void Muscle::do_report_extent(FEM3DExtraConstitution::ReportExtentInfo& info)
{
    info.energy_count(passive_coeffs.size());
    info.gradient_count(passive_coeffs.size() * StencilSize);

    if(info.gradient_only())
        return;

    info.hessian_count(passive_coeffs.size() * HalfHessianSize);
}

void Muscle::do_init(FiniteElementExtraConstitution::FilteredInfo& info)
{
    using ForEachInfo = FiniteElementMethod::ForEachInfo;

    auto geo_slots = world().scene().geometries();

    size_t primitive_count = 0;

    info.for_each(
        geo_slots,
        [&](geometry::SimplicialComplex& sc) -> auto
        {
            primitive_count += sc.tetrahedra().size();
            h_passive_coeffs.resize(primitive_count);
            h_active_coeffs.resize(primitive_count);
            h_directions.resize(primitive_count);

            auto passive   = sc.tetrahedra().find<Float>("passive_modulus");
            auto active    = sc.tetrahedra().find<Float>("active_modulus");
            auto direction = sc.tetrahedra().find<Vector3>("direction");

            return zip(passive->view(), active->view(), direction->view());
        },
        [&](const ForEachInfo& I, auto params)
        {
            auto&& [passive, active, direction] = params;

            auto vI = I.global_index();

            h_passive_coeffs[vI] = passive;
            h_active_coeffs[vI]  = active;
            h_directions[vI]     = direction;
        });

    passive_coeffs.resize(primitive_count);
    passive_coeffs.view().copy_from(h_passive_coeffs.data());

    active_coeffs.resize(primitive_count);
    active_coeffs.view().copy_from(h_active_coeffs.data());

    directions.resize(primitive_count);
    directions.view().copy_from(h_directions.data());
}

void Muscle::do_compute_energy(FEM3DExtraConstitution::ComputeEnergyInfo& info)
{
    using namespace muda;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(info.indices().size(),
                [passive_coeffs = passive_coeffs.cviewer().name("passive_coeffs"),
                active_coeffs = active_coeffs.cviewer().name("active_coeffs"),
                directions = directions.cviewer().name("directions"),
                energies   = info.energies().viewer().name("energies"),
                indices    = info.indices().viewer().name("indices"),
                xs         = info.xs().viewer().name("xs"),
                Dm_invs    = info.Dm_invs().viewer().name("Dm_invs"),
                volumes    = info.rest_volumes().viewer().name("volumes"),
                dt         = info.dt()] __device__(int I)
                {
                    const Vector4i&  tet        = indices(I);
                    const Matrix3x3& Dm_inv     = Dm_invs(I);
                    Float            mu_passive = passive_coeffs(I);
                    Float            mu_active  = active_coeffs(I);
                    Vector3          direction  = directions(I);

                    const Vector3& x0 = xs(tet(0));
                    const Vector3& x1 = xs(tet(1));
                    const Vector3& x2 = xs(tet(2));
                    const Vector3& x3 = xs(tet(3));

                    auto F = fem::F(x0, x1, x2, x3, Dm_inv);

                    Float E_passive, E_active;

                    muscle_passive::E(E_passive, mu_passive, direction, F);
                    muscle_active::E(E_active, mu_active, direction, F);

                    energies(I) = (E_passive + E_active) * dt * dt * volumes(I);
                });
}

void Muscle::do_compute_gradient_hessian(FEM3DExtraConstitution::ComputeGradientHessianInfo& info)
{
    using namespace muda;
    auto gradient_only = info.gradient_only();

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(info.indices().size(),
                [passive_coeffs = passive_coeffs.cviewer().name("passive_coeffs"),
                active_coeffs = active_coeffs.cviewer().name("active_coeffs"),
                directions = directions.cviewer().name("directions"),
                indices    = info.indices().viewer().name("indices"),
                xs         = info.xs().viewer().name("xs"),
                Dm_invs    = info.Dm_invs().viewer().name("Dm_invs"),
                G3s        = info.gradients().viewer().name("gradients"),
                H3x3s      = info.hessians().viewer().name("hessians"),
                volumes    = info.rest_volumes().viewer().name("volumes"),
                dt         = info.dt(),
                gradient_only] __device__(int I) mutable
                {
                    const Vector4i&  tet        = indices(I);
                    const Matrix3x3& Dm_inv     = Dm_invs(I);
                    Float            mu_passive = passive_coeffs(I);
                    Float            mu_active  = active_coeffs(I);
                    Vector3          direction  = directions(I);

                    const Vector3& x0 = xs(tet(0));
                    const Vector3& x1 = xs(tet(1));
                    const Vector3& x2 = xs(tet(2));
                    const Vector3& x3 = xs(tet(3));

                    auto F = fem::F(x0, x1, x2, x3, Dm_inv);

                    auto Vdt2 = volumes(I) * dt * dt;

                    Matrix3x3 dEdF_passive, dEdF_active;
                    muscle_passive::dEdVecF(dEdF_passive, mu_passive, direction, F);
                    muscle_active::dEdVecF(dEdF_active, mu_active, direction, F);
                    auto VecdEdF = flatten(dEdF_passive + dEdF_active);
                    VecdEdF *= Vdt2;

                    Matrix9x12 dFdx = fem::dFdx(Dm_inv);
                    Vector12   G    = dFdx.transpose() * VecdEdF;

                    DoubletVectorAssembler DVA{G3s};
                    DVA.segment<StencilSize>(I * StencilSize).write(tet, G);

                    if(gradient_only)
                        return;

                    Matrix9x9 ddEddF_passive, ddEddF_active;
                    muscle_passive::ddEddVecF(ddEddF_passive, mu_passive, direction, F);
                    muscle_active::ddEddVecF(ddEddF_active, mu_active, direction, F);
                    Matrix9x9 ddEddF = ddEddF_passive + ddEddF_active;
                    make_spd(ddEddF);
                    ddEddF *= Vdt2;
                    Matrix12x12 H = dFdx.transpose() * ddEddF * dFdx;
                    TripletMatrixAssembler TMA{H3x3s};
                    TMA.half_block<StencilSize>(I * HalfHessianSize).write(tet, H);
                });
}

REGISTER_SIM_SYSTEM(Muscle);


class MuscleFEMExporter final : public FEMExporter
{
  public:
    using FEMExporter::FEMExporter;

    SimSystemSlot<FEM3DExtraConstitution> fem_constitution;

    U64 get_uid() const noexcept override
    {
        return Muscle::ConstitutionUID;
    }

    void do_build(BuildInfo&) override
    {
        fem_constitution = require<Muscle>(QueryOptions{.exact = false});
    }

    // ------------------------------------------------------------------
    // Energy  —  one Float per tet, plus tet topology
    //
    // Geometry layout after call:
    //   energy_geo.instances()[t] = { "topo": Vector4i, "energy": Float }
    //   t = 0 … N_tets-1
    // ------------------------------------------------------------------
    void get_fem_energy(geometry::Geometry& energy_geo) override
    {
        auto indices  = fem_constitution->element_indices();   // CBufferView<Vector4i>
        auto energies = fem_constitution->element_energies();  // CBufferView<Float>

        energy_geo.instances().resize(indices.size());

        // Topology
        auto topo = energy_geo.instances().find<Vector4i>("topo");
        if(!topo)
            topo = energy_geo.instances().create<Vector4i>("topo", Vector4i::Zero());
        auto topo_view = view(*topo);
        indices.copy_to(topo_view.data());

        // Per-element energy
        auto energy = energy_geo.instances().find<Float>("energy");
        if(!energy)
            energy = energy_geo.instances().create<Float>("energy", 0.0f);
        auto energy_view = view(*energy);
        energies.copy_to(energy_view.data());
    }

    // ------------------------------------------------------------------
    // Gradient — doublet set: (i: IndexT, grad: Vector3)
    //   N_doublets = 4 * N_tets  (one per tet-corner vertex)
    //   Elastic force on vertex i = -sum of all grad entries with index i
    // ------------------------------------------------------------------
    void get_fem_gradient(geometry::Geometry& grad_geo) override
    {
        auto grads = fem_constitution->element_gradients();  // CDoubletVectorView<Float,3>

        grad_geo.instances().resize(grads.doublet_count());

        auto i = grad_geo.instances().find<IndexT>("i");
        if(!i)
            i = grad_geo.instances().create<IndexT>("i", -1);
        auto i_view = view(*i);
        grads.indices().copy_to(i_view.data());

        auto grad = grad_geo.instances().find<Vector3>("grad");
        if(!grad)
            grad = grad_geo.instances().create<Vector3>("grad", Vector3::Zero());
        auto grad_view = view(*grad);
        grads.values().copy_to(grad_view.data());
    }

    // ------------------------------------------------------------------
    // Hessian — triplet set: (i: IndexT, j: IndexT, hess: Matrix3x3)
    //   N_triplets = 16 * N_tets  (4×4 blocks per tet, off-diagonal included)
    // ------------------------------------------------------------------
    void get_fem_hessian(geometry::Geometry& hess_geo) override
    {
        auto hess = fem_constitution->element_hessians();  // CTripletMatrixView<Float,3>

        hess_geo.instances().resize(hess.triplet_count());

        auto i = hess_geo.instances().find<IndexT>("i");
        if(!i)
            i = hess_geo.instances().create<IndexT>("i", -1);
        auto i_view = view(*i);
        hess.row_indices().copy_to(i_view.data());

        auto j = hess_geo.instances().find<IndexT>("j");
        if(!j)
            j = hess_geo.instances().create<IndexT>("j", -1);
        auto j_view = view(*j);
        hess.col_indices().copy_to(j_view.data());

        auto h = hess_geo.instances().find<Matrix3x3>("hess");
        if(!h)
            h = hess_geo.instances().create<Matrix3x3>("hess", Matrix3x3::Zero());
        auto h_view = view(*h);
        hess.values().copy_to(h_view.data());
    }
};

REGISTER_SIM_SYSTEM(MuscleFEMExporter);


// class MuscleAccessor final : public core::MuscleAccessor, public SimSystem
// {
//   public:
//     using SimSystem::SimSystem;
//     SimSystemSlot<Muscle> m_constitution;

//     void do_build() override
//     {
//         m_constitution = require<Muscle>(QueryOptions{.exact = false});

//         // Register as the world-level feature so that
//         //   world.features().find<core::MuscleController>()
//         // returns this object.
//         auto feature = std::make_shared<core::MuscleController>(this);
//         features().insert(feature);
//     }

//     void do_copy_active_coeffs_from(geometry::SimplicialComplex& geo) override
//     {
//         auto attr = geo.tetrahedra().find<Float>("active_modulus");
//         UIPC_ASSERT(attr,
//                     "[MuscleAccessor] Geometry has no 'active_modulus' attribute.");

//         auto attr_view = view(*attr);
//         UIPC_ASSERT(attr_view.size() == m_constitution->active_coeffs.size(),
//                     "[MuscleAccessor] Size mismatch: "
//                     "geometry has {} entries, device buffer has {}.",
//                     attr_view.size(),
//                     m_constitution->active_coeffs.size());

//         // std::vector<Float> active_moduli;
//         // active_moduli.assign(attr_view.begin(), attr_view.end());
//         m_constitution->active_coeffs.view().copy_from(attr_view.data());
//     }
// };
// REGISTER_SIM_SYSTEM(MuscleAccessor);

}  // namespace uipc::backend::cuda
