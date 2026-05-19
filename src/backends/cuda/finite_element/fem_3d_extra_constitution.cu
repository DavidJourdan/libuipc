#include <finite_element/fem_3d_extra_constitution.h>
#include <finite_element/finite_element_method.h>

namespace uipc::backend::cuda
{
// ──────────────────────────────────────────────────────────────────────────────
// Private bridge overrides
// ──────────────────────────────────────────────────────────────────────────────

void FEM3DExtraConstitution::do_build(FiniteElementExtraConstitution::BuildInfo& info)
{
    // Forward to the 3-D BuildInfo overload — no extra data needed yet.
    FEM3DExtraConstitution::BuildInfo this_info;
    do_build(this_info);
}

void FEM3DExtraConstitution::do_compute_energy(
    FiniteElementExtraConstitution::ComputeEnergyInfo& info)
{
    FEM3DExtraConstitution::ComputeEnergyInfo this_info{this, &info, info.dt()};

    // Capture: muda::BufferView<Float> → muda::CBufferView<Float> (implicit)
    m_element_energies = this_info.energies();
    // Topology is stable; capture once. Subsequent overwrites are harmless.
    m_element_indices = this_info.indices();

    do_compute_energy(this_info);
}

void FEM3DExtraConstitution::do_compute_gradient_hessian(
    FiniteElementExtraConstitution::ComputeGradientHessianInfo& info)
{
    FEM3DExtraConstitution::ComputeGradientHessianInfo this_info{
        this, &info, info.dt()};

    // Capture: DoubletVectorView → CDoubletVectorView (implicit)
    m_element_gradients = this_info.gradients();
    // Only valid when gradient_only == false; exporter must check.
    m_element_hessians = this_info.hessians();

    do_compute_gradient_hessian(this_info);
}

// ──────────────────────────────────────────────────────────────────────────────
// BaseInfo accessors
//
// m_impl is FEM3DExtraConstitution*.
// m_impl->fem()       — protected method added to FiniteElementExtraConstitution,
//                       returns FiniteElementMethod::Impl& (same pattern as
//                       FEM3DConstitution::BaseInfo calling m_impl->fem()).
// m_impl->geo_infos() — already protected in FiniteElementExtraConstitution.
//
// Full buffers are returned (no per-geometry slicing here) because vertex
// indices address the global position array; callers iterate geo_infos() and
// slice manually — exactly as in FEM3DConstitution::BaseInfo.
// ──────────────────────────────────────────────────────────────────────────────

span<const FiniteElementMethod::GeoInfo>
FEM3DExtraConstitution::BaseInfo::geo_infos() const noexcept
{
    return m_impl->geo_infos();
}

muda::CBufferView<Vector3>
FEM3DExtraConstitution::BaseInfo::xs() const noexcept
{
    return m_impl->fem().xs.view();
}

muda::CBufferView<Vector3>
FEM3DExtraConstitution::BaseInfo::x_bars() const noexcept
{
    return m_impl->fem().x_bars.view();
}

muda::CBufferView<Vector4i>
FEM3DExtraConstitution::BaseInfo::indices() const noexcept
{
    return m_impl->fem().tets.view();
}

muda::CBufferView<Matrix3x3>
FEM3DExtraConstitution::BaseInfo::Dm_invs() const noexcept
{
    return m_impl->fem().Dm3x3_invs.view();
}

muda::CBufferView<Float>
FEM3DExtraConstitution::BaseInfo::rest_volumes() const noexcept
{
    return m_impl->fem().rest_volumes.view();
}

muda::CBufferView<Vector4i> FEM3DExtraConstitution::element_indices() const noexcept
{
    return m_element_indices;
}

muda::CBufferView<Float> FEM3DExtraConstitution::element_energies() const noexcept
{
    return m_element_energies;
}

muda::CDoubletVectorView<Float, 3> FEM3DExtraConstitution::element_gradients() const noexcept
{
    return m_element_gradients;
}

muda::CTripletMatrixView<Float, 3> FEM3DExtraConstitution::element_hessians() const noexcept
{
    return m_element_hessians;
}
}  // namespace uipc::backend::cuda