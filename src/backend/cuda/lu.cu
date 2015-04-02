/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <lu.hpp>
#include <err_common.hpp>

#if defined(WITH_LINEAR_ALGEBRA)

#include <cusolverDnManager.hpp>
#include <memory.hpp>
#include <copy.hpp>

#include <math.hpp>
#include <err_common.hpp>

#include <kernel/lu_split.hpp>

namespace cuda
{

//cusolverStatus_t CUDENSEAPI cusolverDn<>getrf_bufferSize(
//        cusolverDnHandle_t handle,
//        int m, int n,
//        <> *A,
//        int lda, int *Lwork );
//
//
//cusolverStatus_t CUDENSEAPI cusolverDn<>getrf(
//        cusolverDnHandle_t handle,
//        int m, int n,
//        <> *A,
//        int lda,
//        <> *Workspace,
//        int *devIpiv, int *devInfo );

template<typename T>
struct getrf_func_def_t
{
    typedef cusolverStatus_t (*getrf_func_def) (
                              cusolverDnHandle_t, int, int,
                              T *, int,
                              T *,
                              int *, int *);
};

template<typename T>
struct getrf_buf_func_def_t
{
    typedef cusolverStatus_t (*getrf_buf_func_def) (
                              cusolverDnHandle_t, int, int,
                              T *, int, int *);
};

#define LU_FUNC_DEF( FUNC )                                                     \
template<typename T>                                                            \
typename FUNC##_func_def_t<T>::FUNC##_func_def                                  \
FUNC##_func();                                                                  \
                                                                                \
template<typename T>                                                            \
typename FUNC##_buf_func_def_t<T>::FUNC##_buf_func_def                          \
FUNC##_buf_func();


#define LU_FUNC( FUNC, TYPE, PREFIX )                                                           \
template<> typename FUNC##_func_def_t<TYPE>::FUNC##_func_def FUNC##_func<TYPE>()              \
{ return &cusolverDn##PREFIX##FUNC; }                                                           \
                                                                                                \
template<> typename FUNC##_buf_func_def_t<TYPE>::FUNC##_buf_func_def FUNC##_buf_func<TYPE>()  \
{ return & cusolverDn##PREFIX##FUNC##_bufferSize; }

LU_FUNC_DEF( getrf )
LU_FUNC(getrf , float  , S)
LU_FUNC(getrf , double , D)
LU_FUNC(getrf , cfloat , C)
LU_FUNC(getrf , cdouble, Z)

template<typename T>
void lu(Array<T> &lower, Array<T> &upper, Array<int> &pivot, const Array<T> &in)
{
    dim4 iDims = in.dims();
    int M = iDims[0];
    int N = iDims[1];

    Array<T> in_copy = copyArray<T>(in);

    int lwork = 0;

    cusolverStatus_t err;
    err = getrf_buf_func<T>()(getSolverHandle(), M, N,
                              in_copy.get(), M, &lwork);

    if(err != CUSOLVER_STATUS_SUCCESS) {
        std::cout <<__PRETTY_FUNCTION__<< " ERROR: " << cusolverErrorString(err) << std::endl;
    }

    T *workspace = memAlloc<T>(lwork);

    pivot = createEmptyArray<int>(af::dim4(min(M, N), 1, 1, 1));
    int *info = memAlloc<int>(1);
    err = getrf_func<T>()(getSolverHandle(), M, N,
                          in_copy.get(), M, workspace,
                          pivot.get(), info);

    if(err != CUSOLVER_STATUS_SUCCESS) {
        std::cout <<__PRETTY_FUNCTION__<< " ERROR: " << cusolverErrorString(err) << std::endl;
    }

    // SPLIT into lower and upper
    dim4 ldims(M, min(M, N));
    dim4 udims(min(M, N), N);
    lower = createEmptyArray<T>(ldims);
    upper = createEmptyArray<T>(udims);

    kernel::lu_split<T>(lower, upper, in_copy);
}

template<typename T>
Array<int> lu_inplace(Array<T> &in)
{
    dim4 iDims = in.dims();
    int M = iDims[0];
    int N = iDims[1];

    Array<int> pivot = createEmptyArray<int>(af::dim4(min(M, N), 1, 1, 1));

    int lwork = 0;

    cusolverStatus_t err;
    err = getrf_buf_func<T>()(getSolverHandle(), M, N,
                              in.get(), M, &lwork);

    if(err != CUSOLVER_STATUS_SUCCESS) {
        std::cout <<__PRETTY_FUNCTION__<< " ERROR: " << cusolverErrorString(err) << std::endl;
    }

    T *workspace = memAlloc<T>(lwork);
    int *info = memAlloc<int>(1);

    err = getrf_func<T>()(getSolverHandle(), M, N,
                          in.get(), M, workspace,
                          pivot.get(), info);

    if(err != CUSOLVER_STATUS_SUCCESS) {
        std::cout <<__PRETTY_FUNCTION__<< " ERROR: " << cusolverErrorString(err) << std::endl;
    }

    return pivot;
}

#define INSTANTIATE_LU(T)                                                                           \
    template Array<int> lu_inplace<T>(Array<T> &in);                                                \
    template void lu<T>(Array<T> &lower, Array<T> &upper, Array<int> &pivot, const Array<T> &in);

INSTANTIATE_LU(float)
INSTANTIATE_LU(cfloat)
INSTANTIATE_LU(double)
INSTANTIATE_LU(cdouble)
}

#else
namespace cuda
{
template<typename T>
void lu(Array<T> &lower, Array<T> &upper, Array<int> &pivot, const Array<T> &in)
{
    AF_ERROR("CUDA cusolver not available. Linear Algebra is disabled",
             AF_ERR_NOT_CONFIGURED);
}

template<typename T>
Array<int> lu_inplace(Array<T> &in)
{
    AF_ERROR("CUDA cusolver not available. Linear Algebra is disabled",
             AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE_LU(T)                                                                           \
    template Array<int> lu_inplace<T>(Array<T> &in);                                                \
    template void lu<T>(Array<T> &lower, Array<T> &upper, Array<int> &pivot, const Array<T> &in);

INSTANTIATE_LU(float)
INSTANTIATE_LU(cfloat)
INSTANTIATE_LU(double)
INSTANTIATE_LU(cdouble)
}
#endif
