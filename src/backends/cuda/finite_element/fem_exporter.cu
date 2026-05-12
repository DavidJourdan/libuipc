#include <finite_element/fem_exporter.h>
#include <finite_element/fem_exporter_manager.h>

namespace uipc::backend::cuda
{
// std::string_view FEMExporter::prim_type() const noexcept
// {
//     return get_prim_type();
// }

U64 FEMExporter::uid() const noexcept
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

void FEMExporter::fem_energy(geometry::Geometry& energy)
{
    get_fem_energy(energy);
}

void FEMExporter::fem_gradient(geometry::Geometry& vert_grad)
{
    get_fem_gradient(vert_grad);
}

void FEMExporter::fem_hessian(geometry::Geometry& vert_hess)
{
    get_fem_hessian(vert_hess);
}
}  // namespace uipc::backend::cuda
