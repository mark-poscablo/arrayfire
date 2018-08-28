/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <Array.hpp>

#include <af/array.h>
#include <af/complex.h>
#include <af/defines.h>
#include <af/lapack.h>
#include <arith.hpp>
#include <blas.hpp>
#include <cast.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <diagonal.hpp>
#include <handle.hpp>
#include <join.hpp>
#include <logic.hpp>
#include <math.hpp>
#include <reduce.hpp>
#include <select.hpp>
#include <svd.hpp>
#include <transpose.hpp>

using af::dim4;
using af::dtype_traits;
using std::vector;
using std::swap;

using namespace detail;

const double dfltTol = 1e-6;

template<typename T>
Array<T> getSubArray(const Array<T> in,
                       uint dim0begin = 0, uint dim0end = 0,
                       uint dim1begin = 0, uint dim1end = 0,
                       uint dim2begin = 0, uint dim2end = 0,
                       uint dim3begin = 0, uint dim3end = 0) {
    vector<af_seq> seqs = {
        {static_cast<double>(dim0begin), static_cast<double>(dim0end), 1.},
        {static_cast<double>(dim1begin), static_cast<double>(dim1end), 1.},
        {static_cast<double>(dim2begin), static_cast<double>(dim2end), 1.},
        {static_cast<double>(dim3begin), static_cast<double>(dim3end), 1.}
    };
    return createSubArray<T>(in, seqs, false);
}

// Moore-Penrose Pseudoinverse
template<typename T>
Array<T> pinverseSvd(const Array<T> &in, const double tol)
{
    in.eval();
    int M = in.dims()[0];
    int N = in.dims()[1];

    // Compute SVD
    typedef typename dtype_traits<T>::base_type Tr;
    Array<Tr> sVec = createEmptyArray<Tr>(dim4(min(M, N)));
    Array<T> u = createEmptyArray<T>(dim4(M, M));
    Array<T> vT = createEmptyArray<T>(dim4(N, N));
    svd<T, Tr>(sVec, u, vT, in);

    // Cast s back to original data type for matmul later
    // (since svd() makes s' type the base type of T)
    Array<T> sVecCast = cast<T, Tr>(sVec);

    Array<T> v = transpose(vT, true);

    // Get reciprocal of sVec's non-zero values for s pinverse, except for
    // very small non-zero values though (< relTol), in order to avoid very
    // large reciprocals
    double relTol = tol * static_cast<double>(max(M, N))
                        * reduce_all<af_max_t, Tr, Tr>(sVec);
    Array<T> relTolArr = createValueArray<T>(sVecCast.dims(), scalar<T>(relTol));
    Array<T> ones = createValueArray<T>(sVecCast.dims(), scalar<T>(1.));
    Array<T> sVecRecip = arithOp<T, af_div_t>(ones, sVecCast, sVecCast.dims());
    Array<char> cond = logicOp<T, af_ge_t>(sVecCast, relTolArr,
                                           sVecCast.dims());
    Array<T> zeros = createValueArray<T>(sVecCast.dims(), scalar<T>(0.));
    sVecRecip = createSelectNode<T>(cond, sVecRecip, zeros, sVecRecip.dims());
    sVecRecip.eval();

    // Make s vector into s pinverse array
    Array<T> sPinv = diagCreate<T>(sVecRecip, 0);

    Array<T> uT = transpose(u, true);

    // Crop v and u* for final matmul later based on s+'s size, because
    // sVec produced by svd() has minimal dim length (no extra zeroes).
    // Thus s+ produced by diagCreate() will have minimal dims as well,
    // and v could have an extra dim0 or u* could have an extra dim1
    if (v.dims()[1] > sPinv.dims()[0]) {
        v = getSubArray(v, 0, v.dims()[0] - 1, 0, sPinv.dims()[0] - 1);
    }
    if (uT.dims()[0] > sPinv.dims()[1]) {
        uT = getSubArray(uT, 0, sPinv.dims()[1] - 1, 0, uT.dims()[1] - 1);
    }

    Array<T> out = matmul<T>(matmul<T>(v, sPinv, AF_MAT_NONE, AF_MAT_NONE),
                             uT, AF_MAT_NONE, AF_MAT_NONE);

    return out;
}

// Naive batching for now
template<typename T>
Array<T> batchedPinverse(const Array<T> &in, const double tol) {
    uint dim2 = in.dims()[2];
    uint dim3 = in.dims()[3];

    if (in.ndims() <= 2) {
        return pinverseSvd<T>(in, tol);
    }
    else {
        vector<Array<T> > finalOutputs;
        for (int j = 0; j < dim3; ++j) {
            vector<Array<T> > outputs;
            for (int i = 0; i < dim2; ++i) {
                vector<af_seq> seqs = {
                    {0., static_cast<double>(in.dims()[0] - 1), 1.},
                    {0., static_cast<double>(in.dims()[1] - 1), 1.},
                    {static_cast<double>(i), static_cast<double>(i), 1.},
                    {static_cast<double>(j), static_cast<double>(j), 1.}
                };
                Array<T> inSlice = createSubArray<T>(in, seqs);
                outputs.push_back(pinverseSvd<T>(inSlice, tol));
            }
            Array<T> mergedOuts = join<T>(2, outputs);
            finalOutputs.push_back(mergedOuts);
        }
        Array<T> finalMergedOuts = join<T>(3, finalOutputs);
        return finalMergedOuts;
    }
}

template<typename T>
static inline af_array pinverse(const af_array in, const double tol)
{
    return getHandle(batchedPinverse<T>(getArray<T>(in), tol));
}

af_err af_pinverse(af_array *out, const af_array in, const double tol,
                   const af_mat_prop options)
{
    try {
        const ArrayInfo& i_info = getInfo(in);

        af_dtype type = i_info.getType();

        if (options != AF_MAT_NONE) {
            AF_ERROR("Using this property is not yet supported in inverse", AF_ERR_NOT_SUPPORTED);
        }

        ARG_ASSERT(1, i_info.isFloating()); // Only floating and complex types
        ARG_ASSERT(2, tol >= 0.); // Ensure tolerance is not negative

        af_array output;

        if(i_info.ndims() == 0) {
            return af_retain_array(out, in);
        }

        switch(type) {
            case f32: output = pinverse<float  >(in, tol);  break;
            case f64: output = pinverse<double >(in, tol);  break;
            case c32: output = pinverse<cfloat >(in, tol);  break;
            case c64: output = pinverse<cdouble>(in, tol);  break;
            default:  TYPE_ERROR(1, type);
        }
        swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

