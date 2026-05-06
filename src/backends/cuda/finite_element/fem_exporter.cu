#include <finite_element/fem_exporter.h>
#include <finite_element/fem_exporter_manager.h>

namespace uipc::backend::cuda
{
// std::string_view FEMExporter::prim_type() const noexcept
// {
//     return get_prim_type();
// }

std::string_view FEMExporter::uid() const noexcept
{
    return get_uid();
}

void FEMExporter::do_build()
{
    auto& manager = require<FEMExporterManager>();

    BuildInfo info;
    do_build(info);

    manager.add_exporter(this);
}

void FEMExporter::fem_energy(std::string_view uid, geometry::Geometry& energy)
{
    get_fem_energy(uid, energy);
}

void FEMExporter::fem_gradient(std::string_view uid, geometry::Geometry& vert_grad)
{
    get_fem_gradient(uid, vert_grad);
}

void FEMExporter::fem_hessian(std::string_view uid, geometry::Geometry& vert_hess)
{
    get_fem_hessian(uid, vert_hess);
}
}  // namespace uipc::backend::cuda
