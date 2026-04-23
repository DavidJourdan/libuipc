// template <typename T>
// __device__ __host__ T sgn(T val) {
//     return (T(0) < val) - (val < T(0));
// }

template <typename T>
__host__ __device__ void E(T& R, const T& mu, const Eigen::Matrix<T, 3, 1>& a, const Eigen::Matrix<T, 3, 3>& F)
{
    auto I5 = (F * a).squaredNorm();
    R = 0.5 * mu * std::pow(std::sqrt(I5) - 1, 2);
}

template <typename T>
__host__ __device__ void dEdVecF(Eigen::Matrix<T, 3, 3>& R, const T& mu, const Eigen::Matrix<T, 3, 1>& a, const Eigen::Matrix<T, 3, 3>& F)
{
    auto I5 = (F * a).squaredNorm();

    R = mu * (1 - 1 / std::sqrt(I5)) * F * (a * a.transpose());
}

template <typename T>
__host__ __device__ void ddEddVecF(Eigen::Matrix<T, 9, 9>& R, const T& mu, const Eigen::Matrix<T, 3, 1>& a, const Eigen::Matrix<T, 3, 3>& F)
{
    auto I5 = (F * a).squaredNorm();

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

    R = mu * ((1 - 1 / std::sqrt(I5)) * H5 + std::pow(I5, -3/2) * fa * fa.transpose());
}
