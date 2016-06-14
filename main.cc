#include <iostream>
#include "ml_base/utils.h"
#include "ml_base/linear_regression.h"

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

    return 0;
}