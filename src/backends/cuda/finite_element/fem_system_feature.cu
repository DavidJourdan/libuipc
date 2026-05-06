#include <finite_element/fem_system_feature.h>
#include <finite_element/fem_exporter_manager.h>

namespace uipc::backend::cuda
{
FEMSystemFeatureOverrider::FEMSystemFeatureOverrider(FEMExporterManager* manager)
{
    UIPC_ASSERT(manager, "Exporter must not be null");
    m_manager = *manager;
}

vector<std::string> FEMSystemFeatureOverrider::get_fem_primitive_types() const
{
    return m_manager->get_fem_primitive_types();
}

void FEMSystemFeatureOverrider::get_fem_gradient(std::string_view prim_type,
                                                         geometry::Geometry& vert_grad)
{
    m_manager->get_fem_gradient(prim_type, vert_grad);
}

void FEMSystemFeatureOverrider::get_fem_hessian(std::string_view prim_type,
                                                        geometry::Geometry& vert_hess)
{
    m_manager->get_fem_hessian(prim_type, vert_hess);
}
void FEMSystemFeatureOverrider::get_fem_energy(std::string_view prim_type,
                                                       geometry::Geometry& prims)
{
    m_manager->get_fem_energy(prim_type, prims);
}
}  // namespace uipc::backend::cuda