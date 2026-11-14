#pragma once
#include <cuda_tool/eigen.h>
#include <finite_element/matrix_utils.h>

namespace uipc::backend::cuda
{
namespace sym::stable_anisotropic_arap
{
#include "detail/stable_anisotropic_arap.inl"
}
}  // namespace uipc::backend::cuda
