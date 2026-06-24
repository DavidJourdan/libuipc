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

    U64 get_uid() const noexcept override { return ConstitutionUID; }
    void do_build(BuildInfo& info) override {}
    void do_report_extent(ReportExtentInfo& info) override;
    void do_init(FiniteElementExtraConstitution::FilteredInfo& info) override;
    void do_compute_energy(ComputeEnergyInfo& info) override;
    void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override;
};

}  // namespace uipc::backend::cuda
