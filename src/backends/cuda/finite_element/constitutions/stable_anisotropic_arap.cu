#include <finite_element/fem_3d_extra_constitution.h>
#include <finite_element/constitutions/stable_anisotropic_arap_function.h>
#include <finite_element/fem_utils.h>
#include <cuda_tool/cuda_tool.h>
#include <Eigen/Dense>
#include <utils/make_spd.h>
#include <utils/matrix_assembler.h>

namespace uipc::backend::cuda
{


namespace
{
    constexpr SizeT StencilSize     = 4;
    constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    namespace SAA = sym::stable_anisotropic_arap;

    __global__ void StableAnisotropicARAP_do_compute_energy_kernel(
        cuda_tool::CBufferView<Float>     mus,
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
        const Vector4i&  tet                = indices(I);
        const Matrix3x3& Dm_inv             = Dm_invs(I);
        Float            anisotropy_modulus = mus(I);
        Vector3          direction          = directions(I);

        const Vector3& x0 = xs(tet(0));
        const Vector3& x1 = xs(tet(1));
        const Vector3& x2 = xs(tet(2));
        const Vector3& x3 = xs(tet(3));

        auto F = fem::F(x0, x1, x2, x3, Dm_inv);

        Float E;

        SAA::E(E, anisotropy_modulus, direction, F);
        E *= dt * dt * volumes(I);
        energies(I) = E;
    }

    __global__ void StableAnisotropicARAP_do_compute_gradient_hessian_kernel(
        cuda_tool::CBufferView<Float>           mus,
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
        const Vector4i&  tet                = indices(I);
        const Matrix3x3& Dm_inv             = Dm_invs(I);
        Float            anisotropy_modulus = mus(I);
        Vector3          direction          = directions(I);

        const Vector3& x0 = xs(tet(0));
        const Vector3& x1 = xs(tet(1));
        const Vector3& x2 = xs(tet(2));
        const Vector3& x3 = xs(tet(3));

        auto F = fem::F(x0, x1, x2, x3, Dm_inv);

        auto Vdt2 = volumes(I) * dt * dt;

        Matrix3x3 dEdF;
        SAA::dEdVecF(dEdF, anisotropy_modulus, direction, F);
        auto VecdEdF = flatten(dEdF);
        VecdEdF *= Vdt2;

        Matrix9x12 dFdx = fem::dFdx(Dm_inv);
        Vector12   G    = dFdx.transpose() * VecdEdF;

        DoubletVectorAssembler DVA{G3s};
        DVA.segment<StencilSize>(I * StencilSize).write(tet, G);

        if(gradient_only)
            return;

        Matrix9x9 ddEddF;
        SAA::ddEddVecF(ddEddF, anisotropy_modulus, direction, F);
        ddEddF *= Vdt2;
        make_spd(ddEddF);
        Matrix12x12            H = dFdx.transpose() * ddEddF * dFdx;
        TripletMatrixAssembler TMA{H3x3s};
        TMA.half_block<StencilSize>(I * HalfHessianSize).write(tet, H);
    }
}  // namespace


class StableAnisotropicARAP final : public FEM3DExtraConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64   ConstitutionUID = 33;
    static constexpr SizeT StencilSize     = 4;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    using FEM3DExtraConstitution::FEM3DExtraConstitution;

    vector<Float>   h_mus;
    vector<Vector3> h_directions;

    cuda_tool::DeviceBuffer<Float>   mus;
    cuda_tool::DeviceBuffer<Vector3> directions;

    virtual U64 get_uid() const noexcept override { return ConstitutionUID; }

    virtual void do_build(BuildInfo& info) override {}

    virtual void do_report_extent(ReportExtentInfo& info) override
    {
        info.energy_count(mus.size());
        info.gradient_count(mus.size() * StencilSize);

        if(info.gradient_only())
            return;

        info.hessian_count(mus.size() * HalfHessianSize);
    }

    virtual void do_init(FiniteElementExtraConstitution::FilteredInfo& info) override
    {
        using ForEachInfo = FiniteElementMethod::ForEachInfo;

        auto geo_slots = world().scene().geometries();

        size_t primitive_count = 0;

        info.for_each(
            geo_slots,
            [&](geometry::SimplicialComplex& sc) -> auto
            {
                primitive_count += sc.tetrahedra().size();
                h_mus.resize(primitive_count);
                h_directions.resize(primitive_count);

                auto mu = sc.tetrahedra().find<Float>("anisotropy_modulus");
                auto direction = sc.tetrahedra().find<Vector3>("direction");

                return zip(mu->view(), direction->view());
            },
            [&](const ForEachInfo& I, auto mu_and_direction)
            {
                auto&& [mu, direction] = mu_and_direction;

                auto vI = I.global_index();

                h_mus[vI]        = mu;
                h_directions[vI] = direction;
            });

        mus.resize(primitive_count);
        mus.view().copy_from(h_mus.data());

        directions.resize(primitive_count);
        directions.view().copy_from(h_directions.data());
    }


    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        auto k = StableAnisotropicARAP_do_compute_energy_kernel;
        int  n = (int)info.indices().size();
        if(n > 0)
        {
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                mus.cview(),
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

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        auto k = StableAnisotropicARAP_do_compute_gradient_hessian_kernel;
        int  n = (int)info.indices().size();
        if(n > 0)
        {
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                mus.cview(),
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
};

REGISTER_SIM_SYSTEM(StableAnisotropicARAP);
}  // namespace uipc::backend::cuda
