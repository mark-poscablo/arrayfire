/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>

using af::array;
using af::matmul;
using af::pinverse;
using af::randu;

// See https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Definition
// for descriptions of these tests

TEST(Pinverse, Cond1) {
    int rows = 4;
    int cols = 3;

    array in = randu(rows, cols);
    array in_pinv = pinverse(in);
    array out = matmul(in, in_pinv, in);
    ASSERT_ARRAYS_NEAR(in, out, 0.001f);
}

TEST(Pinverse, Cond2) {
    int rows = 4;
    int cols = 3;

    array in = randu(rows, cols);
    array in_pinv = pinverse(in);
    array out = matmul(in_pinv, in, in_pinv);
    ASSERT_ARRAYS_NEAR(in_pinv, out, 0.001f);
}

TEST(Pinverse, Cond3) {
    int rows = 4;
    int cols = 3;

    array in = randu(rows, cols);
    array in_pinv = pinverse(in);
    array aaplus = matmul(in, in_pinv);
    array out = matmul(in, in_pinv).H();
    ASSERT_ARRAYS_NEAR(aaplus, out, 0.001f);
}

TEST(Pinverse, Cond4) {
    int rows = 4;
    int cols = 3;

    array in = randu(rows, cols);
    array in_pinv = pinverse(in);
    af::array aplusa = af::matmul(in_pinv, in);
    af::array out = af::matmul(in_pinv, in).H();
    ASSERT_ARRAYS_NEAR(aplusa, out, 0.001f);
}

