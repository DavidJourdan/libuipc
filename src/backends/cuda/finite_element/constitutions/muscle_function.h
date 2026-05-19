#pragma once
#include <muda/ext/eigen/eigen_core_cxx20.h>
#include <finite_element/matrix_utils.h>
#include <finite_element/fem_utils.h>

namespace uipc::backend::cuda
{
namespace muscle_passive
{
template <typename T>
__host__ __device__ void E(T& R, const T& mu, const Eigen::Matrix<T, 3, 1>& a, const Eigen::Matrix<T, 3, 3>& F)
{
    auto I5 = (F * a).squaredNorm();
    auto I4 = uipc::backend::cuda::fem::invariant4(F, a);

    if(I4 >= 1)
        R = 1 / 3. * mu * std::pow(std::sqrt(I5) - 1, 3);
    else
        R = 0;
}

template <typename T>
__host__ __device__ void dEdVecF(Eigen::Matrix<T, 3, 3>& R, const T& mu, const Eigen::Matrix<T, 3, 1>& a, const Eigen::Matrix<T, 3, 3>& F)
{
    auto I5 = (F * a).squaredNorm();
    auto I4 = uipc::backend::cuda::fem::invariant4(F, a);

    if(I4 >= 1)
        R = mu * std::pow(std::sqrt(I5) - 1, 2) / std::sqrt(I5) * F * (a * a.transpose());
    else
        R = Eigen::Matrix<T, 3, 3>::Zero();
}

template <typename T>
__host__ __device__ void ddEddVecF(Eigen::Matrix<T, 9, 9>& R, const T& mu, const Eigen::Matrix<T, 3, 1>& a, const Eigen::Matrix<T, 3, 3>& F)
{
    auto I5 = (F * a).squaredNorm();
    auto I4 = uipc::backend::cuda::fem::invariant4(F, a);

    Eigen::Matrix<T, 3, 3> A = a * a.transpose();
    Eigen::Matrix<T, 9, 9> H5;
    H5.block<3, 3>(0, 0) = A(0, 0) * Eigen::Matrix<T, 3, 3>::Identity();
    H5.block<3, 3>(0, 3*1) = A(0, 1) * Eigen::Matrix<T, 3, 3>::Identity();
    H5.block<3, 3>(0, 3*2) = A(0, 2) * Eigen::Matrix<T, 3, 3>::Identity();
    H5.block<3, 3>(3*1, 0) = A(1, 0) * Eigen::Matrix<T, 3, 3>::Identity();
    H5.block<3, 3>(3*1, 3*1) = A(1, 1) * Eigen::Matrix<T, 3, 3>::Identity();
    H5.block<3, 3>(3*1, 3*2) = A(1, 2) * Eigen::Matrix<T, 3, 3>::Identity();
    H5.block<3, 3>(3*2, 0) = A(2, 0) * Eigen::Matrix<T, 3, 3>::Identity();
    H5.block<3, 3>(3*2, 3*1) = A(2, 1) * Eigen::Matrix<T, 3, 3>::Identity();
    H5.block<3, 3>(3*2, 3*2) = A(2, 2) * Eigen::Matrix<T, 3, 3>::Identity();

    auto fa = flatten(F * A);

    if(I4 >= 1)
        R = mu / std::sqrt(I5) * (std::pow(std::sqrt(I5) - 1, 2) * H5 + (1 - 1 / I5) * fa * fa.transpose());
    else
        R = Eigen::Matrix<T, 9, 9>::Zero();
}
}

namespace muscle_active
{
template <typename T>
__host__ __device__ void E(T& R, const T& mu, const Eigen::Matrix<T, 3, 1>& a, const Eigen::Matrix<T, 3, 3>& F)
{
    T sqrtI5 = (F * a).norm();
    auto I4 = uipc::backend::cuda::fem::invariant4(F, a);

    if(I4 < 0.4)
        R = 0;
    else if(I4 < 0.6)
        R = 3 * std::pow(sqrtI5, 3) - 3.6 * std::pow(sqrtI5, 2) + 1.44 * sqrtI5 - 0.192;
    else if(I4 < 1.4)
        R = -4 / 3. * std::pow(sqrtI5, 3) + 4 * std::pow(sqrtI5, 2) - 3 * sqrtI5 + 0.672;
    else if(I4 < 1.6)
        R = 3 * std::pow(sqrtI5, 3) - 14.4 * std::pow(sqrtI5, 2) + 23.04 * sqrtI5 - 11.592 - 0.056 / 3;
    else
        R = 2.032 / 3;

    R *= mu;
}

template <typename T>
__host__ __device__ void dEdVecF(Eigen::Matrix<T, 3, 3>& R, const T& mu, const Eigen::Matrix<T, 3, 1>& a, const Eigen::Matrix<T, 3, 3>& F)
{
    T sqrtI5 = (F * a).norm();
    auto I4 = uipc::backend::cuda::fem::invariant4(F, a);

    if(I4 < 0.4)
        R = Eigen::Matrix<T, 3, 3>::Zero();
    else if(I4 < 0.6)
        R = 9 * std::pow(sqrtI5 - 0.4, 2) / sqrtI5 * F * (a * a.transpose());
    else if(I4 < 1.4)
        R = (1 - 4 * std::pow(sqrtI5 - 1, 2)) / sqrtI5 * F * (a * a.transpose());
    else if(I4 < 1.6)
        R = 9 * std::pow(sqrtI5 - 1.6, 2) / sqrtI5 * F * (a * a.transpose());
    else
        R = Eigen::Matrix<T, 3, 3>::Zero();

    R *= mu;
}

template <typename T>
__host__ __device__ void ddEddVecF(Eigen::Matrix<T, 9, 9>& R, const T& mu, const Eigen::Matrix<T, 3, 1>& a, const Eigen::Matrix<T, 3, 3>& F)
{
    auto I5 = (F * a).squaredNorm();
    auto I4 = uipc::backend::cuda::fem::invariant4(F, a);

    Eigen::Matrix<T, 3, 3> A = a * a.transpose();
    Eigen::Matrix<T, 9, 9> H5;
    H5.block<3, 3>(0, 0) = A(0, 0) * Eigen::Matrix<T, 3, 3>::Identity();
    H5.block<3, 3>(0, 3*1) = A(0, 1) * Eigen::Matrix<T, 3, 3>::Identity();
    H5.block<3, 3>(0, 3*2) = A(0, 2) * Eigen::Matrix<T, 3, 3>::Identity();
    H5.block<3, 3>(3*1, 0) = A(1, 0) * Eigen::Matrix<T, 3, 3>::Identity();
    H5.block<3, 3>(3*1, 3*1) = A(1, 1) * Eigen::Matrix<T, 3, 3>::Identity();
    H5.block<3, 3>(3*1, 3*2) = A(1, 2) * Eigen::Matrix<T, 3, 3>::Identity();
    H5.block<3, 3>(3*2, 0) = A(2, 0) * Eigen::Matrix<T, 3, 3>::Identity();
    H5.block<3, 3>(3*2, 3*1) = A(2, 1) * Eigen::Matrix<T, 3, 3>::Identity();
    H5.block<3, 3>(3*2, 3*2) = A(2, 2) * Eigen::Matrix<T, 3, 3>::Identity();

    auto fa = flatten(F * A);

    if(I4 < 0.4)
        R = Eigen::Matrix<T, 9, 9>::Zero();
    else if(I4 < 0.6)
        R = 9 * (std::pow(std::sqrt(I5) - 0.4, 2) / std::sqrt(I5) * H5 + (2 * (std::sqrt(I5) - 0.4) / I5 - std::pow(I5, -3/2) * std::pow(std::sqrt(I5) - 0.4, 2)) * fa * fa.transpose());
    else if(I4 < 1.4)
        R = (1 - 4 * (std::pow(std::sqrt(I5) - 1, 2))) / std::sqrt(I5) * H5 + (-8 * (std::sqrt(I5) - 1) / I5 - std::pow(I5, -3/2) * (1 - 4 * std::pow(std::sqrt(I5) - 1, 2))) * fa * fa.transpose();
    else if(I4 < 1.6)
        R = 9 * (std::pow(std::sqrt(I5) - 1.6, 2) / std::sqrt(I5) * H5 + (2 * (std::sqrt(I5) - 1.6) / I5 - std::pow(I5, -3/2) * std::pow(std::sqrt(I5) - 1.6, 2)) * fa * fa.transpose());
    else
        R = Eigen::Matrix<T, 9, 9>::Zero();

    R *= mu;
}
}
}  // namespace uipc::backend::cuda
