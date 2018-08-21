/*******************************************************
 * Copyright (c) 2014, ArrayFire
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
using std::vector;
using namespace detail;

const double dfltTol = 1e-6;

template<typename T>
Array<T> pinverseSvd(const Array<T> &in, const double tol)
{
    // Moore-Penrose Pseudoinverse

    in.eval();
    int M = in.dims()[0];
    int N = in.dims()[1];

    // Compute SVD
    typedef typename af::dtype_traits<T>::base_type Tr;
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
        std::vector<af_seq> seqs = {
            {0., static_cast<double>(v.dims()[0] - 1), 1.},
            {0., static_cast<double>(sPinv.dims()[0] - 1), 1.}
        };
        v = createSubArray<T>(v, seqs);
    }
    if (uT.dims()[0] > sPinv.dims()[1]) {
        std::vector<af_seq> seqs = {
            {0., static_cast<double>(sPinv.dims()[1] - 1), 1.},
            {0., static_cast<double>(uT.dims()[1] - 1), 1.}
        };
        uT = createSubArray<T>(uT, seqs);
    }

    Array<T> out = matmul<T>(matmul<T>(v, sPinv, AF_MAT_NONE, AF_MAT_NONE),
                             uT, AF_MAT_NONE, AF_MAT_NONE);

    return out;
}

template<typename T>
static inline af_array pinverse(const af_array in, const double tol)
{
    Array<T> inArray = getArray<T>(in);
    uint batchSize = inArray.dims()[2];

    if (inArray.ndims() < 3) {
        return getHandle(pinverseSvd<T>(getArray<T>(in), tol));
    }
    else {
        vector<Array<T> > outputs;
        for (int i = 0; i < batchSize; ++i) {
            vector<af_seq> seqs = {
                {0., static_cast<double>(inArray.dims()[0] - 1), 1.},
                {0., static_cast<double>(inArray.dims()[1] - 1), 1.},
                {static_cast<double>(i), static_cast<double>(i), 1.}
            };
            Array<T> inSlice = createSubArray<T>(inArray, seqs);
            outputs.push_back(pinverseSvd<T>(inSlice, tol));
        }
        Array<T> mergedOuts = join<T>(2, outputs);
        return getHandle(mergedOuts);
    }
}

af_err af_pinverse(af_array *out, const af_array in, const double tol,
                   const af_mat_prop options)
{
    try {
        const ArrayInfo& i_info = getInfo(in);

        if (i_info.ndims() > 3) {
            AF_ERROR("solve can not be used in batch mode", AF_ERR_BATCH);
        }

        af_dtype type = i_info.getType();

        if (options != AF_MAT_NONE) {
            AF_ERROR("Using this property is not yet supported in inverse", AF_ERR_NOT_SUPPORTED);
        }

        ARG_ASSERT(1, i_info.isFloating()); // Only floating and complex types
        ARG_ASSERT(2, tol >= 0.); // Only floating and complex types

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
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

