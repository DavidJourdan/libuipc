#pragma once
#include <muda/ext/eigen/eigen_core_cxx20.h>
#include <finite_element/matrix_utils.h>

namespace uipc::backend::cuda
{
namespace sym::stable_anisotropic_arap
{
#include "detail/stable_anisotropic_arap.inl"
}
}  // namespace uipc::backend::cuda
