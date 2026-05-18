#include <fmt/ranges.h>
#include <finite_element/fem_exporter_manager.h>
#include <finite_element/fem_system_feature.h>
#include <finite_element/fem_exporter.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(FEMExporterManager);

void FEMExporterManager::do_build()
{
    auto overrider = std::make_shared<FEMSystemFeatureOverrider>(this);
    auto feature   = std::make_shared<core::FEMSystemFeature>(overrider);
    features().insert(feature);

    on_init_scene([&] { init(); });
}

void FEMExporterManager::init()
{
    auto exporters = m_exporters.view();
    m_uids.resize(exporters.size());

    for(auto&& [i, exporter] : enumerate(exporters))
    {
        U64 uid = exporter->uid();
        auto it  = m_exporter_map.find(uid);
        if(it != m_exporter_map.end())
        {
            logger::warn("FEM exporter for constitution uid '{}'<{}> already exists, overwriting with <{}>.",
                         uid,
                         it->second->name(),
                         exporter->name());
            it->second = exporter;
        }
        else
        {
            m_exporter_map.emplace(uid, exporter);
        }

        m_uids[i] = uid;
    }
}

void FEMExporterManager::get_fem_energy(U64 uid, geometry::Geometry& prim_energy)
{
    auto exporter = find_exporter(uid);
    if(!exporter)
        return;
    exporter->fem_energy(prim_energy);
}

void FEMExporterManager::get_fem_gradient(U64 uid, geometry::Geometry& prim_grad)
{
    auto exporter = find_exporter(uid);
    if(!exporter)
        return;
    exporter->fem_gradient(prim_grad);
}

void FEMExporterManager::get_fem_hessian(U64 uid, geometry::Geometry& prim_hess)
{
    auto exporter = find_exporter(uid);
    if(!exporter)
        return;
    exporter->fem_hessian(prim_hess);
}

void FEMExporterManager::add_exporter(FEMExporter* exporter)
{
    UIPC_ASSERT(exporter, "Exporter must not be null");
    check_state(SimEngineState::BuildSystems, "add_exporter");
    m_exporters.register_sim_system(*exporter);
}

FEMExporter* FEMExporterManager::find_exporter(U64 uid) const
{
    auto it = m_exporter_map.find(uid);
    if(it != m_exporter_map.end())
    {
        return it->second;
    }
    else
    {
        logger::warn(R"(FEM exporter for primitive type '{}' not found
Supported types are: [{}])",
                     uid,
                     fmt::join(m_uids, ", "));

        return nullptr;
    }
}

}  // namespace uipc::backend::cuda
