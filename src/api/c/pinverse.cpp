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
#include <logic.hpp>
#include <math.hpp>
#include <select.hpp>
#include <svd.hpp>
#include <transpose.hpp>

using af::dim4;
using namespace detail;

template<typename T>
Array<T> pinverse_svd(const Array<T> &in)
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

    Array<T> v = transpose(vT, true);

    // Round down small values to zero to avoid large reciprocals later
    Array<Tr> eps = createValueArray<Tr>(sVec.dims(), scalar<Tr>(1e-6));
    Array<char> cond = logicOp<Tr, af_lt_t>(sVec, eps, sVec.dims());
    Array<Tr> sVecRoundOff = createSelectNode<Tr, true>(cond, sVec, 0., sVec.dims());

    // Generate s+
    Array<Tr> ones = createValueArray<Tr>(sVecRoundOff.dims(), scalar<Tr>(1.));
    Array<Tr> sVecRecip = arithOp<Tr, af_div_t>(ones, sVecRoundOff, sVecRoundOff.dims());
    Array<Tr> sPinv = diagCreate<Tr>(sVecRecip, 0);

    // Cast s+ back to original data type for matmul later
    // (since svd makes s' type the base type of T)
    Array<T> sPinvCast = cast<T, Tr>(sPinv);

    Array<T> uT = transpose(u, true);

    // Adjust v and u* for final matmul, based on s+'s size
    // Recall that in+ = v s+ u*
    // sVec produced by af::svd() has minimal dim length (no extra zero)
    // Therefore, s+ produced by af::diag() will have minimal dims as well
    //  (no extra zeroed dim0 or dim1)
    // v's dim1 must == s+'s dim0, so if it's >, the last dim1 should be removed
    // u*'s dim0 must == s+'s dim1, so if it's >, the last dim0 should be removed
    // Removal of extra dim0/dim1 doesn't affect integrity of computation because
    //  extra dim0/dim1 will theoretically be just multiplied with s+'s zero dim1/dim0
    if (v.dims()[1] > sPinvCast.dims()[0]) {
        std::vector<af_seq> seqs = {
            af_span,
            {0., static_cast<double>(sPinvCast.dims()[0]), 1.}
        };
        v = createSubArray<T>(v, seqs);
    }
    if (uT.dims()[0] > sPinvCast.dims()[1]) {
        std::vector<af_seq> seqs = {
            {0., static_cast<double>(sPinvCast.dims()[1]), 1.},
            af_span
        };
        uT = createSubArray<T>(uT, seqs);
    }

    Array<T> out = matmul<T>(matmul<T>(v, sPinvCast, AF_MAT_NONE, AF_MAT_NONE),
                             uT, AF_MAT_NONE, AF_MAT_NONE);

    return out;
}

template<typename T>
static inline af_array pinverse(const af_array in)
{
    return getHandle(pinverse_svd<T>(getArray<T>(in)));
}

af_err af_pinverse(af_array *out, const af_array in, const af_mat_prop options)
{
    try {
        const ArrayInfo& i_info = getInfo(in);

        if (i_info.ndims() > 2) {
            AF_ERROR("solve can not be used in batch mode", AF_ERR_BATCH);
        }

        af_dtype type = i_info.getType();

        if (options != AF_MAT_NONE) {
            AF_ERROR("Using this property is not yet supported in inverse", AF_ERR_NOT_SUPPORTED);
        }

        ARG_ASSERT(1, i_info.isFloating()); // Only floating and complex types

        af_array output;

        if(i_info.ndims() == 0) {
            return af_retain_array(out, in);
        }

        switch(type) {
            case f32: output = pinverse<float  >(in);  break;
            case f64: output = pinverse<double >(in);  break;
            case c32: output = pinverse<cfloat >(in);  break;
            case c64: output = pinverse<cdouble>(in);  break;
            default:  TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

