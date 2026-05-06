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
        auto uid = std::string{exporter->uid()};
        auto it        = m_exporter_map.find(uid);
        if(it != m_exporter_map.end())
        {
            logger::warn("FEM exporter for primitive type '{}'<{}> already exists, overwriting with <{}>.",
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

void FEMExporterManager::get_fem_energy(std::string_view    uid,
                                                geometry::Geometry& prim_energy)
{
    auto exporter = find_exporter(uid);
    if(!exporter)
        return;
    exporter->fem_energy(uid, prim_energy);
}

void FEMExporterManager::get_fem_gradient(std::string_view    uid,
                                                  geometry::Geometry& prim_grad)
{
    auto exporter = find_exporter(uid);
    if(!exporter)
        return;
    exporter->fem_gradient(uid, prim_grad);
}

void FEMExporterManager::get_fem_hessian(std::string_view    uid,
                                                 geometry::Geometry& prim_hess)
{
    auto exporter = find_exporter(uid);
    if(!exporter)
        return;
    exporter->fem_hessian(uid, prim_hess);
}

// vector<std::string> FEMExporterManager::get_fem_primitive_types() const
// {
//     return m_fem_prim_types;
// }

void FEMExporterManager::add_exporter(FEMExporter* exporter)
{
    UIPC_ASSERT(exporter, "Exporter must not be null");
    check_state(SimEngineState::BuildSystems, "add_exporter");
    m_exporters.register_sim_system(*exporter);
}

FEMExporter* FEMExporterManager::find_exporter(std::string_view uid) const
{
    auto it = m_exporter_map.find(std::string{uid});
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

// void FEMExporterManager::_create_prim_type_on_geo(std::string_view prim_type_v,
//                                                       geometry::Geometry& geo)
// {
//     auto prim_type = geo.meta().find<std::string>("prim_type");
//     if(!prim_type)
//     {
//         prim_type = geo.meta().create<std::string>("prim_type");
//     }
//     view(*prim_type)[0] = prim_type_v;
// }
}  // namespace uipc::backend::cuda
