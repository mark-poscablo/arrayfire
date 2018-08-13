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

template<typename T>
void pinverseTester(const int m, const int n, double eps)
{
    if (noDoubleTests<T>()) return;
    if (noLAPACKTests()) return;
#if 1
    array A  = cpu_randu<T>(dim4(m, n));
#else
    array A  = randu(m, n, (dtype)dtype_traits<T>::af_type);
#endif

    //! [ex_inverse]
    array IA = pinverse(A);
    array I = matmul(A, IA);
    //! [ex_inverse]

    array I2 = identity(m, m, (dtype)dtype_traits<T>::af_type);

    ASSERT_ARRAYS_NEAR(I2, I, eps);
}


template<typename T>
class Pinverse : public ::testing::Test
{

};

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

TYPED_TEST(Pinverse, Regular) {
    pinverseTester<TypeParam>(1000, 800, eps<TypeParam>());
}

TYPED_TEST(Pinverse, MultiplePowerOfTwo) {
    pinverseTester<TypeParam>(2048, 1024, eps<TypeParam>());
}
// See https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Definition
// for descriptions of these tests

const int rows = 1000;
const int cols = 800;
TEST(Pinverse, Cond1) {
    array in = randu(rows, cols);
    array in_pinv = pinverse(in);
    array out = matmul(in, in_pinv, in);
    ASSERT_ARRAYS_NEAR(in, out, 0.001f);
}

TEST(Pinverse, Cond2) {
    array in = randu(rows, cols);
    array in_pinv = pinverse(in);
    array out = matmul(in_pinv, in, in_pinv);
    ASSERT_ARRAYS_NEAR(in_pinv, out, 0.001f);
}

TEST(Pinverse, Cond3) {
    array in = randu(rows, cols);
    array in_pinv = pinverse(in);
    array aaplus = matmul(in, in_pinv);
    array out = matmul(in, in_pinv).H();
    ASSERT_ARRAYS_NEAR(aaplus, out, 0.001f);
}

TEST(Pinverse, Cond4) {
    array in = randu(rows, cols);
    array in_pinv = pinverse(in);
    af::array aplusa = af::matmul(in_pinv, in);
    af::array out = af::matmul(in_pinv, in).H();
    ASSERT_ARRAYS_NEAR(aplusa, out, 0.001f);
}

