#include <finite_element/fem_3d_constitution.h>

namespace uipc::backend::cuda
{
void FEM3DConstitution::do_build(FiniteElementConstitution::BuildInfo& info)
{
    FEM3DConstitution::BuildInfo this_info;
    do_build(this_info);
}

void FEM3DConstitution::do_compute_energy(FiniteElementConstitution::ComputeEnergyInfo& info)
{
    FEM3DConstitution::ComputeEnergyInfo this_info{
        this, m_index_in_dim, info.dt(), info.energies()};

    // Capture: muda::BufferView<Float> → muda::CBufferView<Float> (implicit)
    m_element_energies = this_info.energies();
    // Topology is stable; capture once. Subsequent overwrites are harmless.
    m_element_indices = this_info.indices();

    do_compute_energy(this_info);
}

void FEM3DConstitution::do_compute_gradient_hessian(FiniteElementConstitution::ComputeGradientHessianInfo& info)
{
    FEM3DConstitution::ComputeGradientHessianInfo this_info{
        this, m_index_in_dim, info.gradient_only(), info.dt(), info.gradients(), info.hessians()};

    // Capture: DoubletVectorView → CDoubletVectorView (implicit)
    m_element_gradients = this_info.gradients();
    // Only valid when gradient_only == false; exporter must check.
    m_element_hessians = this_info.hessians();

    do_compute_gradient_hessian(this_info);
}

IndexT FEM3DConstitution::get_dim() const noexcept
{
    return 3;
}

muda::CBufferView<Vector4i> FEM3DConstitution::BaseInfo::indices() const noexcept
{
    auto& info = constitution_info();
    return m_impl->fem().tets.view(info.primitive_offset, info.primitive_count);
}

muda::CBufferView<Vector3> FEM3DConstitution::BaseInfo::xs() const noexcept
{
    return m_impl->fem().xs.view();  // must return full buffer, because the indices index into the full buffer
}

muda::CBufferView<Vector3> FEM3DConstitution::BaseInfo::x_bars() const noexcept
{
    return m_impl->fem().x_bars.view();  // must return full buffer, because the indices index into the full buffer
}

muda::CBufferView<Matrix3x3> FEM3DConstitution::BaseInfo::Dm_invs() const noexcept
{
    auto& info = constitution_info();
    return m_impl->fem().Dm3x3_invs.view(info.primitive_offset, info.primitive_count);
}

muda::CBufferView<Float> FEM3DConstitution::BaseInfo::rest_volumes() const noexcept
{
    auto& info = constitution_info();
    return m_impl->fem().rest_volumes.view(info.primitive_offset, info.primitive_count);
}

const FiniteElementMethod::ConstitutionInfo& FEM3DConstitution::BaseInfo::constitution_info() const noexcept
{
    return m_impl->constitution_info();
}

muda::CBufferView<Vector4i> FEM3DConstitution::element_indices() const noexcept
{
    return m_element_indices;
}

muda::CBufferView<Float> FEM3DConstitution::element_energies() const noexcept
{
    return m_element_energies;
}

muda::CDoubletVectorView<Float, 3> FEM3DConstitution::element_gradients() const noexcept
{
    return m_element_gradients;
}

muda::CTripletMatrixView<Float, 3> FEM3DConstitution::element_hessians() const noexcept
{
    return m_element_hessians;
}
}  // namespace uipc::backend::cuda
