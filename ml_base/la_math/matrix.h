//
// Created by g8y3e on 6/16/16.
//

#ifndef ML_LINEAR_REGRESSION_MATRIX_H
#define ML_LINEAR_REGRESSION_MATRIX_H

#include <cstdio>

#include <vector>
#include <string>

namespace ml_base {
namespace la_math {
    class MatrixException : public std::exception {
    public:
        MatrixException(const char* message)
            : message_(message) {
        }

    public:
         virtual const char* what() const _NOEXCEPT {
             return message_.c_str();
         }

    private:
        std::string message_;
    };

    template <typename T>
    class Matrix {
    public:
        typedef std::vector<std::vector<T>> matrix_type;

    public:
        Matrix(size_t rows, size_t columns)
                : rows_(rows), columns_(columns) {
            data_.reserve(columns_);

            for (size_t i = 0; i < columns_; ++i) {
                data_.push_back(std::vector<T>(rows_));
            }
        }
        ~Matrix() {
        }

    public:
        std::vector<T>& operator[](size_t index) {
            return data_[index];
        }

        const std::vector<T>& operator[](size_t index) const {
            return data_[index];
        }

    public:
        const Matrix<T>& operator=(const Matrix<T>& rhs);

    public:
        size_t  getRows() const {
            return rows_;
        }

        size_t getColumns() const {
            return columns_;
        }

    private:
        matrix_type data_;

        size_t rows_;
        size_t columns_;
    };
}
}

#endif //ML_LINEAR_REGRESSION_MATRIX_H
