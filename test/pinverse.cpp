/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/traits.hpp>
#include <iostream>
#include <complex>
#include <testHelpers.hpp>

using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype;
using af::dtype_traits;
using af::identity;
using af::matmul;
using af::max;
using af::pinverse;
using af::randu;
using std::abs;
using std::string;
using std::vector;

template<typename T>
array makeComplex(dim4 dims, const vector<T>& real, const vector<T>& imag) {
    array realArr(dims, &real.front());
    array imagArr(dims, &imag.front());
    return af::complex(realArr, imagArr);
}

template<typename T>
array readTestInput(string testFilePath) {
    typedef typename dtype_traits<T>::base_type InBaseType;
    dtype outAfType = (dtype) dtype_traits<T>::af_type;

    vector<dim4> dimsVec;
    vector<vector<InBaseType> > inVec;
    vector<vector<InBaseType> > goldVec;
    readTestsFromFile<InBaseType, InBaseType>(testFilePath, dimsVec, inVec, goldVec);
    dim4 inDims = dimsVec[0];

    if (outAfType == c32 || outAfType == c64) {
        return makeComplex(inDims, inVec[1], inVec[2]);
    }
    else {
        return array(inDims, &inVec[0].front());
    }
}

template<typename T>
array readTestGold(string testFilePath) {
    typedef typename dtype_traits<T>::base_type InBaseType;
    dtype outAfType = (dtype) dtype_traits<T>::af_type;

    vector<dim4> dimsVec;
    vector<vector<InBaseType> > inVec;
    vector<vector<InBaseType> > goldVec;
    readTestsFromFile<InBaseType, InBaseType>(testFilePath, dimsVec, inVec, goldVec);
    dim4 goldDims(dimsVec[0][1], dimsVec[0][0]);

    if (outAfType == c32 || outAfType == c64) {
        return makeComplex(goldDims, goldVec[1], goldVec[2]);
    }
    else {
        return array(goldDims, &goldVec[0].front());
    }
}

template<typename T>
class Pinverse : public ::testing::Test
{

};

// Epsilons taken from test/inverse.cpp
template<typename T>
double eps();

template<>
double eps<float>() {
    return 0.01f;
}

template<>
double eps<double>() {
    return 1e-5;
}

template<>
double eps<cfloat>() {
    return 0.01f;
}

template<>
double eps<cdouble>() {
    return 1e-5;
}

typedef ::testing::Types<float, cfloat, double, cdouble> TestTypes;
TYPED_TEST_CASE(Pinverse, TestTypes);

// Test Moore-Penrose conditions
// See https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Definition

TYPED_TEST(Pinverse, AApinvA_A) {
    array in = readTestInput<TypeParam>(string(TEST_DIR"/pinverse/pinverse10x8.test"));
    array inpinv = pinverse(in);
    array out = matmul(in, inpinv, in);
    ASSERT_ARRAYS_NEAR(in, out, eps<TypeParam>());
}

TYPED_TEST(Pinverse, ApinvAApinv_Apinv) {
    array in = readTestInput<TypeParam>(string(TEST_DIR"/pinverse/pinverse10x8.test"));
    array inpinv = pinverse(in);
    array out = matmul(inpinv, in, inpinv);
    ASSERT_ARRAYS_NEAR(inpinv, out, eps<TypeParam>());
}

TYPED_TEST(Pinverse, AApinv_IsHermitian) {
    array in = readTestInput<TypeParam>(string(TEST_DIR"/pinverse/pinverse10x8.test"));
    array inpinv = pinverse(in);
    array aapinv = matmul(in, inpinv);
    array out = matmul(in, inpinv).H();
    ASSERT_ARRAYS_NEAR(aapinv, out, eps<TypeParam>());
}

TYPED_TEST(Pinverse, ApinvA_IsHermitian) {
    array in = readTestInput<TypeParam>(string(TEST_DIR"/pinverse/pinverse10x8.test"));
    array inpinv = pinverse(in);
    array apinva = af::matmul(inpinv, in);
    array out = af::matmul(inpinv, in).H();
    ASSERT_ARRAYS_NEAR(apinva, out, eps<TypeParam>());
}

TEST(Pinverse, CustomTol) {
    array in = readTestInput<float>(string(TEST_DIR"/pinverse/pinverse10x8.test"));
    array inpinv = pinverse(in, 1e-12);
    array out = matmul(in, inpinv, in);
    ASSERT_ARRAYS_NEAR(in, out, eps<float>());
}

TEST(Pinverse, C) {
    array in = readTestInput<float>(string(TEST_DIR"/pinverse/pinverse10x8.test"));
    af_array inpinv = 0, out = 0;
    af_pinverse(&inpinv, in.get(), -1, AF_MAT_NONE);
    af_matmul(&out, in.get(), inpinv, AF_MAT_NONE, AF_MAT_NONE);
    af_matmul(&out, out, in.get(), AF_MAT_NONE, AF_MAT_NONE);
    ASSERT_ARRAYS_NEAR(in.get(), out, eps<float>());
}

TEST(Pinverse, C_CustomTol) {
    array in = readTestInput<float>(string(TEST_DIR"/pinverse/pinverse10x8.test"));
    af_array inpinv = 0, out = 0;
    af_pinverse(&inpinv, in.get(), 1e-12, AF_MAT_NONE);
    af_matmul(&out, in.get(), inpinv, AF_MAT_NONE, AF_MAT_NONE);
    af_matmul(&out, out, in.get(), AF_MAT_NONE, AF_MAT_NONE);
    ASSERT_ARRAYS_NEAR(in.get(), out, eps<float>());
}

TEST(Pinverse, CompareWithNumpy) {
    array in = readTestInput<float>(string(TEST_DIR"/pinverse/pinverse10x8.test"));
    array gold = readTestGold<float>(string(TEST_DIR"/pinverse/pinverse10x8.test"));
    array out = pinverse(in);
    ASSERT_ARRAYS_NEAR(gold, out, eps<float>());
}

TEST(Pinverse, Square) {
    array in = readTestInput<float>(string(TEST_DIR"/pinverse/pinverse10x10.test"));
    array inpinv = pinverse(in);
    array out = matmul(in, inpinv, in);
    ASSERT_ARRAYS_NEAR(in, out, eps<float>());
}

TEST(Pinverse, Dim1GtDim0) {
    array in = readTestInput<float>(string(TEST_DIR"/pinverse/pinverse8x10.test"));
    array inpinv = pinverse(in);
    array out = matmul(in, inpinv, in);
    ASSERT_ARRAYS_NEAR(in, out, eps<float>());
}

TEST(Pinverse, SmallSigValExists) {
    array in = readTestInput<float>(string(TEST_DIR"/pinverse/pinverse10x8.test"));
    const dim_t dim0 = in.dims(0);
    const dim_t dim1 = in.dims(1);

    // Generate sigma with small non-zero value
    af::array u;
    af::array vT;
    af::array sVec;
    af::svd(u, sVec, vT, in);
    dim_t sSize = sVec.elements();

    sVec(2) = 1e-10;
    af::array s = af::diag(sVec, 0, false);
    af::array zeros = af::constant(0,
                                   dim0 > sSize ? dim0 - sSize : sSize,
                                   dim1 > sSize ? dim1 - sSize : sSize);
    s = af::join(dim0 > dim1 ? 0 : 1, s, zeros);

    // Make new input array that has a small non-zero value in its SVD sigma
    in = af::matmul(u, s, vT);
    array inpinv = pinverse(in, 1e-6);
    array out = matmul(in, inpinv, in);
    ASSERT_ARRAYS_NEAR(in, out, eps<float>());
}

