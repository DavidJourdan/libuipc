#include <uipc/core/fem_system_feature.h>

namespace uipc::core
{
FEMSystemFeature::FEMSystemFeature(S<FEMSystemFeatureOverrider> overrider)
    : m_impl(std::move(overrider))
{
}

void FEMSystemFeature::fem_energy(std::string_view prim_type, geometry::Geometry& prims)
{
    m_impl->get_fem_energy(prim_type, prims);
}

void FEMSystemFeature::fem_gradient(std::string_view    prim_type,
                                            geometry::Geometry& vert_grad)
{
    m_impl->get_fem_gradient(prim_type, vert_grad);
}

void FEMSystemFeature::fem_hessian(std::string_view    prim_type,
                                           geometry::Geometry& vert_hess)
{
    m_impl->get_fem_hessian(prim_type, vert_hess);
}

void FEMSystemFeature::fem_energy(const constitution::IConstitution& c,
                                          geometry::Geometry& prims)
{
    auto uid_str = fmt::format("#{}", c.uid());
    m_impl->get_fem_energy(uid_str, prims);
}

void FEMSystemFeature::fem_gradient(const constitution::IConstitution& c,
                                            geometry::Geometry& vert_grad)
{
    auto uid_str = fmt::format("#{}", c.uid());
    m_impl->get_fem_gradient(uid_str, vert_grad);
}

void FEMSystemFeature::fem_hessian(const constitution::IConstitution& c,
                                           geometry::Geometry& vert_hess)
{
    auto uid_str = fmt::format("#{}", c.uid());
    m_impl->get_fem_hessian(uid_str, vert_hess);
}

vector<std::string> FEMSystemFeature::fem_primitive_types() const
{
    return m_impl->get_fem_primitive_types();
}

std::string_view FEMSystemFeature::get_name() const
{
    return FeatureName;
}
}  // namespace uipc::core