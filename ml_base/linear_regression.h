//
// Created by g8y3e on 6/14/16.
//

#ifndef ML_LINEAR_REGRESSION_LINEARREGRESSION_H
#define ML_LINEAR_REGRESSION_LINEARREGRESSION_H

#include <utility>
#include <string>
#include <list>

namespace ml_base {
    class LinearRegression {
    public:
        LinearRegression();
        ~LinearRegression();

    public:
        void addDataFromFile(std::string data_path);
        void addData(double first_data, double second_data);

    public:
        void calculate();

    public:
        double predicFirstData(double second_data);
        double predictSecondData(double first_data);

    private:
        double getCostFunctionValue(double first_param, double second_param);
        double getAvgSumm(double first_param, double second_param, bool is_first = true);

    private:
        std::list<std::pair<double, double>> data_;

        double first_param_;
        double second_param_;
    };
}

#endif //ML_LINEAR_REGRESSION_LINEARREGRESSION_H
