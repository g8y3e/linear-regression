#include <iostream>
#include "ml_base/utils.h"
#include "ml_base/linear_regression.h"
#include "ml_base/la_math/matrix.h"
#include "ml_base/la_math/matrix_operation.h"

using namespace std;

int main(int argc, char **argv) {
    std::string file_path = "data/example_data";
    if (true) {
        file_path = "/Users/valentine.pavchuk/g8y3e/study/ml-linear-regression/" + file_path;
    }

    ml_base::LinearRegression linear_regression;

    linear_regression.addDataFromFile(file_path);
    linear_regression.calculate();

    std::cout << "predict x: " << linear_regression.predicFirstData(1) << std::endl;
    std::cout << "predict y: " << linear_regression.predictSecondData(42) << std::endl;

    // test summ
    ml_base::la_math::Matrix<int> test1(3, 2);
    test1[0][0] = 1;    test1[1][0] = 1;
    test1[0][1] = 1;    test1[1][1] = 1;
    test1[0][2] = 1;    test1[1][2] = 1;

    ml_base::la_math::Matrix<int> test2(3, 2);
    test2[0][0] = 41;   test2[1][0] = 39;
    test2[0][1] = 40;   test2[1][1] = 38;
    test2[0][2] = 1;    test2[1][2] = 1;


    ml_base::la_math::Matrix<int> test3 = test1 + test2;

    // test mult
    ml_base::la_math::Matrix<int> test4(2, 2);
    test4[0][0] = 2;    test4[1][0] = 0;
    test4[0][1] = 1;    test4[1][1] = 2;

    ml_base::la_math::Matrix<int> test5 = test1 * test4;

    // transpose test
    ml_base::la_math::Matrix<int> testTr1(3, 2);
    testTr1[0][0] = 1;   testTr1[1][0] = 4;
    testTr1[0][1] = 2;   testTr1[1][1] = 5;
    testTr1[0][2] = 3;   testTr1[1][2] = 6;

    ml_base::la_math::Matrix<int> testTr2 = transpose(testTr1);
    ml_base::la_math::Matrix<int> testTr3 = transpose(testTr2);

    // determinant test
    int determinantRes1 = determinant(test4);

    ml_base::la_math::Matrix<int> testDet(4, 4);
    testDet[0][0] =  3;   testDet[1][0] = -3;    testDet[2][0] = -5;    testDet[3][0] =  8;
    testDet[0][1] = -3;   testDet[1][1] =  2;    testDet[2][1] =  4;    testDet[3][1] = -6;
    testDet[0][2] =  2;   testDet[1][2] = -5;    testDet[2][2] = -7;    testDet[3][2] =  5;
    testDet[0][3] = -4;   testDet[1][3] =  3;    testDet[2][3] =  5;    testDet[3][3] = -6;
    int determinantRes2 = determinant(testDet);

    // test inverse
    ml_base::la_math::Matrix<float> testInverse(2, 2);
    testInverse[0][0] = 1;   testInverse[1][0] = 4;
    testInverse[0][1] = 2;   testInverse[1][1] = 5;

    float deterTest2 = determinant(testInverse);

    ml_base::la_math::Matrix<float> resultInverse = inverse(testInverse);
    ml_base::la_math::Matrix<float> resultInvTest = testInverse * resultInverse;

    return 0;
}