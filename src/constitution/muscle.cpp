#include <uipc/constitution/muscle.h>
#include <uipc/builtin/constitution_uid_auto_register.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/builtin/constitution_type.h>
#include <uipc/constitution/conversion.h>
#include <uipc/common/log.h>

namespace uipc::constitution
{
REGISTER_CONSTITUTION_UIDS()
{
    using namespace uipc::builtin;
    list<UIDInfo> uids;
    uids.push_back(UIDInfo{.uid = 999, .name = "Muscle", .type = string{builtin::FiniteElement}});
    return uids;
}

Muscle::Muscle(const Json& config) noexcept
    : m_config(config)
{
}

void Muscle::apply_to(geometry::SimplicialComplex& sc,
                      Vector3                      direction,
                      Float                        passive_modulus,
                      Float                        active_modulus) const
{
    Base::apply_to(sc);

    UIPC_ASSERT(sc.dim() == 3, "Muscle only supports 3D simplicial complex");

    auto passive_attr = sc.tetrahedra().find<Float>("passive_modulus");
    if(!passive_attr)
        passive_attr = sc.tetrahedra().create<Float>("passive_modulus", passive_modulus);
    std::ranges::fill(geometry::view(*passive_attr), passive_modulus);

    auto active_attr = sc.tetrahedra().find<Float>("active_modulus");
    if(!active_attr)
        active_attr = sc.tetrahedra().create<Float>("active_modulus", active_modulus);
    std::ranges::fill(geometry::view(*active_attr), active_modulus);

    auto direction_attr = sc.tetrahedra().find<Vector3>("direction");
    if(!direction_attr)
        direction_attr = sc.tetrahedra().create<Vector3>("direction", direction);
    std::ranges::fill(geometry::view(*direction_attr), direction);
}

Json Muscle::default_config() noexcept
{
    return Json::object();
}

U64 Muscle::get_uid() const noexcept
{
    return 999;
}
}  // namespace uipc::constitution
