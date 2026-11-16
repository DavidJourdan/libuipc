#include <finite_element/constitutions/muscle.h>
#include <finite_element/constitutions/muscle_function.h>
#include <finite_element/fem_3d_extra_constitution.h>
#include <finite_element/fem_utils.h>
#include <cuda_tool/cuda_tool.h>
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

namespace
{
    constexpr SizeT StencilSize     = 4;
    constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    __global__ void Muscle_do_compute_energy_kernel(
        cuda_tool::CBufferView<Float>     passive_coeffs,
        cuda_tool::CBufferView<Float>     active_coeffs,
        cuda_tool::CBufferView<Vector3>   directions,
        cuda_tool::BufferView<Float>      energies,
        cuda_tool::CBufferView<Vector4i>  indices,
        cuda_tool::CBufferView<Vector3>   xs,
        cuda_tool::CBufferView<Matrix3x3> Dm_invs,
        cuda_tool::CBufferView<Float>     volumes,
        Float                             dt,
        int                               n)
    {
        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;
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
    }

    __global__ void Muscle_do_compute_gradient_hessian_kernel(
        cuda_tool::CBufferView<Float>           passive_coeffs,
        cuda_tool::CBufferView<Float>           active_coeffs,
        cuda_tool::CBufferView<Vector3>         directions,
        cuda_tool::CBufferView<Vector4i>        indices,
        cuda_tool::CBufferView<Vector3>         xs,
        cuda_tool::CBufferView<Matrix3x3>       Dm_invs,
        cuda_tool::DoubletVectorView<Float, 3>  G3s,
        cuda_tool::TripletMatrixView<Float, 3>   H3x3s,
        cuda_tool::CBufferView<Float>            volumes,
        Float                                    dt,
        bool                                     gradient_only,
        int                                      n)
    {
        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;
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
        Matrix12x12            H = dFdx.transpose() * ddEddF * dFdx;
        TripletMatrixAssembler TMA{H3x3s};
        TMA.half_block<StencilSize>(I * HalfHessianSize).write(tet, H);
    }
}  // namespace

void Muscle::do_compute_energy(FEM3DExtraConstitution::ComputeEnergyInfo& info)
{
    auto k = Muscle_do_compute_energy_kernel;
    int  n = (int)info.indices().size();
    if(n > 0)
    {
        k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            passive_coeffs.cview(),
            active_coeffs.cview(),
            directions.cview(),
            info.energies(),
            info.indices(),
            info.xs(),
            info.Dm_invs(),
            info.rest_volumes(),
            info.dt(),
            n);
    }
}

void Muscle::do_compute_gradient_hessian(FEM3DExtraConstitution::ComputeGradientHessianInfo& info)
{
    auto k = Muscle_do_compute_gradient_hessian_kernel;
    int  n = (int)info.indices().size();
    if(n > 0)
    {
        k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            passive_coeffs.cview(),
            active_coeffs.cview(),
            directions.cview(),
            info.indices(),
            info.xs(),
            info.Dm_invs(),
            info.gradients(),
            info.hessians(),
            info.rest_volumes(),
            info.dt(),
            info.gradient_only(),
            n);
    }
}

REGISTER_SIM_SYSTEM(Muscle);


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
