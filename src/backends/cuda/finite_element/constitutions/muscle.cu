#include <finite_element/fem_3d_extra_constitution.h>
#include <finite_element/constitutions/muscle_function.h>
#include <finite_element/fem_utils.h>
#include <Eigen/Dense>
#include <utils/make_spd.h>
#include <utils/matrix_assembler.h>

namespace uipc::backend::cuda
{
class Muscle final : public FEM3DExtraConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64   ConstitutionUID = 999;
    static constexpr SizeT StencilSize     = 4;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    using FEM3DExtraConstitution::FEM3DExtraConstitution;

    vector<Float>   h_passive_coeffs;
    vector<Float>   h_active_coeffs;
    vector<Vector3> h_directions;

    muda::DeviceBuffer<Float>   passive_coeffs;
    muda::DeviceBuffer<Float>   active_coeffs;
    muda::DeviceBuffer<Vector3> directions;

    virtual U64 get_uid() const noexcept override { return ConstitutionUID; }

    virtual void do_build(BuildInfo& info) override {}

    virtual void do_report_extent(ReportExtentInfo& info) override
    {
        info.energy_count(passive_coeffs.size());
        info.gradient_count(passive_coeffs.size() * StencilSize);

        if(info.gradient_only())
            return;

        info.hessian_count(passive_coeffs.size() * HalfHessianSize);
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

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
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

                       Float E;

                       muscle_passive::E(E, mu_passive, direction, F);
                       E *= dt * dt * volumes(I);
                       energies(I) = E;
                   });
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
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

                       Matrix3x3 dEdF;
                       muscle_passive::dEdVecF(dEdF, mu_passive, direction, F);
                       auto VecdEdF = flatten(dEdF);
                       VecdEdF *= Vdt2;

                       Matrix9x12 dFdx = fem::dFdx(Dm_inv);
                       Vector12   G    = dFdx.transpose() * VecdEdF;

                       DoubletVectorAssembler DVA{G3s};
                       DVA.segment<StencilSize>(I * StencilSize).write(tet, G);

                       if(gradient_only)
                           return;

                       Matrix9x9 ddEddF;
                       muscle_passive::ddEddVecF(ddEddF, mu_passive, direction, F);
                       ddEddF *= Vdt2;
                       make_spd(ddEddF);
                       Matrix12x12 H = dFdx.transpose() * ddEddF * dFdx;
                       TripletMatrixAssembler TMA{H3x3s};
                       TMA.half_block<StencilSize>(I * HalfHessianSize).write(tet, H);
                   });
    }
};

REGISTER_SIM_SYSTEM(Muscle);
}  // namespace uipc::backend::cuda
