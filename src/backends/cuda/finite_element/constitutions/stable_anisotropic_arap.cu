#include <finite_element/fem_3d_extra_constitution.h>
#include <finite_element/constitutions/stable_anisotropic_arap_function.h>
#include <finite_element/fem_utils.h>
#include <Eigen/Dense>
#include <utils/make_spd.h>
#include <utils/matrix_assembler.h>

namespace uipc::backend::cuda
{
class StableAnisotropicARAP final : public FEM3DExtraConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64   ConstitutionUID = 33;
    static constexpr SizeT StencilSize     = 4;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    using FEM3DExtraConstitution::FEM3DExtraConstitution;

    vector<Float> h_mus;
    vector<Vector3> h_directions;

    muda::DeviceBuffer<Float> mus;
    muda::DeviceBuffer<Vector3> directions;

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

                auto mu        = sc.tetrahedra().find<Float>("anisotropy_modulus");
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
        using namespace muda;
        namespace SAA = sym::stable_anisotropic_arap;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.indices().size(),
                   [mus        = mus.cviewer().name("mus"),
                    directions = directions.cviewer().name("directions"),
                    energies   = info.energies().viewer().name("energies"),
                    indices    = info.indices().viewer().name("indices"),
                    xs         = info.xs().viewer().name("xs"),
                    Dm_invs    = info.Dm_invs().viewer().name("Dm_invs"),
                    volumes    = info.rest_volumes().viewer().name("volumes"),
                    dt         = info.dt()] __device__(int I)
                   {
                       const Vector4i&  tet       = indices(I);
                       const Matrix3x3& Dm_inv    = Dm_invs(I);
                       Float            anisotropy_modulus        = mus(I);
                       Vector3          direction = directions(I);

                       const Vector3& x0 = xs(tet(0));
                       const Vector3& x1 = xs(tet(1));
                       const Vector3& x2 = xs(tet(2));
                       const Vector3& x3 = xs(tet(3));

                       auto F = fem::F(x0, x1, x2, x3, Dm_inv);

                       Float E;

                       SAA::E(E, anisotropy_modulus, direction, F);
                       E *= dt * dt * volumes(I);
                       energies(I) = E;
                   });
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;
        namespace SAA      = sym::stable_anisotropic_arap;
        auto gradient_only = info.gradient_only();

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.indices().size(),
                   [mus        = mus.cviewer().name("mus"),
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
                       const Vector4i&  tet       = indices(I);
                       const Matrix3x3& Dm_inv    = Dm_invs(I);
                       Float            anisotropy_modulus        = mus(I);
                       Vector3          direction = directions(I);

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
                       Matrix12x12 H = dFdx.transpose() * ddEddF * dFdx;
                       TripletMatrixAssembler TMA{H3x3s};
                       TMA.half_block<StencilSize>(I * HalfHessianSize).write(tet, H);
                   });
    }
};

REGISTER_SIM_SYSTEM(StableAnisotropicARAP);
}  // namespace uipc::backend::cuda
