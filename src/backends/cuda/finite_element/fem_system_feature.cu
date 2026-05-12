#include <finite_element/fem_system_feature.h>
#include <finite_element/fem_exporter_manager.h>

namespace uipc::backend::cuda
{
FEMSystemFeatureOverrider::FEMSystemFeatureOverrider(FEMExporterManager* manager)
{
    UIPC_ASSERT(manager, "Exporter must not be null");
    m_manager = *manager;
}

void FEMSystemFeatureOverrider::get_fem_gradient(U64 uid, geometry::Geometry& vert_grad)
{
    m_manager->get_fem_gradient(uid, vert_grad);
}

void FEMSystemFeatureOverrider::get_fem_hessian(U64 uid, geometry::Geometry& vert_hess)
{
    m_manager->get_fem_hessian(uid, vert_hess);
}
void FEMSystemFeatureOverrider::get_fem_energy(U64 uid, geometry::Geometry& prims)
{
    m_manager->get_fem_energy(uid, prims);
}
}  // namespace uipc::backend::cuda