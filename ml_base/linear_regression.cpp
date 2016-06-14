//
// Created by g8y3e on 6/14/16.
//

#include "linear_regression.h"

#include <sstream>
#include <cmath>

#include "utils.h"

namespace ml_base {
    /**
     *
     */
    LinearRegression::LinearRegression() {
    }

    LinearRegression::~LinearRegression() {
    }

    /**
     * Adding data to object from file
     * @param data_path Path to file with data.
     */
    void LinearRegression::addDataFromFile(std::string data_path) {
        auto data_list = ReadFileFromPath(data_path);

        for (auto data_str : data_list) {
            std::istringstream in_string_stream(data_str);

            double first_data, second_data;
            if (!(in_string_stream >> first_data >> second_data)) {
                continue;
            }

            data_.push_back(std::make_pair(first_data, second_data));
        }
    }

    /**
     * Adding data to object from parameters
     * @param first_data Parameter data.
     * @param second_data Parameter data.
     */
    void LinearRegression::addData(double first_data, double second_data) {
        for (auto data_pair : data_) {
            if (data_pair.first == first_data && data_pair.second == second_data) {
                return;
            }
        }

        data_.push_back(std::make_pair(first_data, second_data));
    }

    /**
     * Calculating linear function
     */
    void LinearRegression::calculate() {
        double first_param = 0.0;
        double second_param = 0.0;
        double learning_step = 1;

        double prev_min_cost = this->getCostFunctionValue(first_param, second_param);

        bool founded_local_min = false;
        bool is_went_minimum = false;

        int iterations = 0;
        while (!founded_local_min) {
            double first_summ = this->getAvgSumm(first_param, second_param);
            double second_summ = this->getAvgSumm(first_param, second_param, false);

            double new_first_param = first_param - learning_step * first_summ;
            double new_second_param = second_param - learning_step * second_summ;

            double current_min_cost = this->getCostFunctionValue(new_first_param, new_second_param);
            ++iterations;
            if (prev_min_cost > current_min_cost) {
                first_param = new_first_param;
                second_param = new_second_param;
                if (!is_went_minimum) {
                    learning_step *= 2;
                }
                is_went_minimum = false;
            } else if (prev_min_cost < current_min_cost) {
                is_went_minimum = true;
                learning_step = -(learning_step / 2.0);
                continue;
            }

            founded_local_min = prev_min_cost == current_min_cost;
            prev_min_cost = current_min_cost;
        }

        this->first_param_ = first_param;
        this->second_param_ = second_param;
    }

    double LinearRegression::getCostFunctionValue(double first_param, double second_param) {
        double sum = 0.0;

        for (auto data_pair : data_) {
            sum += pow((first_param + second_param * data_pair.first) - data_pair.second, 2);
        }

        return (sum / static_cast<double>(data_.size()));
    }

    double LinearRegression::getAvgSumm(double first_param, double second_param, bool is_first) {
        double sum = 0.0;

        for (auto data_pair : data_) {
          double entry_value = (first_param + second_param * data_pair.first) - data_pair.second;
          sum += (is_first) ? entry_value : entry_value * data_pair.first;
        }

        return (sum / static_cast<double>(data_.size()));
    }

    /**
     * Make an prediction based on second parameter data
     * @param second_data Parameter data.
     */
    double LinearRegression::predicFirstData(double second_data) {
        return (second_data - this->first_param_) / this->second_param_;
    }

    /**
    * Make an prediction based on first parameter data
    * @param first_data Parameter data.
    */
    double LinearRegression::predictSecondData(double first_data) {
        return this->first_param_ + this->second_param_ * first_data;
    }
}