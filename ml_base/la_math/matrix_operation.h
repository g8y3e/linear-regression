//
// Created by Valentine.Pavchuk on 6/16/16.
//

#ifndef ML_LINEAR_REGRESSION_MATRIX_OPERATION_H
#define ML_LINEAR_REGRESSION_MATRIX_OPERATION_H

#include "matrix.h"

namespace ml_base {
namespace la_math {
    template <typename T>
    Matrix<T> operator+(const Matrix<T>& lhs, const Matrix<T>& rhs) {
        if (rhs.getRows() != lhs.getRows() || rhs.getColumns() != lhs.getColumns()) {
            throw MatrixException("Matrix can't add not same dimension matrices!");
        }

        Matrix<T> result(lhs.getRows(), lhs.getColumns());

        for (size_t j = 0; j < lhs.getColumns(); ++j) {
            for (size_t i = 0; i < lhs.getRows(); ++i) {
                result[j][i] = lhs[j][i] + rhs[j][i];
            }
        }

        return result;
    }

    template <typename T>
    Matrix<T> operator-(const Matrix<T>& lhs, const Matrix<T>& rhs) {
        if (rhs.getRows() != lhs.getRows() || rhs.getColumns() != lhs.getColumns()) {
            throw MatrixException("Matrix can't sub not same dimension matrices!");
        }

        Matrix<T> result(lhs.getRows(), lhs.getColumns());

        for (size_t j = 0; j < lhs.getColumns(); ++j) {
            for (size_t i = 0; i < lhs.getRows(); ++i) {
                result[j][i] = lhs[j][i] - rhs[j][i];
            }
        }

        return result;
    }

    template <typename T>
    Matrix<T> operator*(const Matrix<T>& lhs, const T& scalar) {
        Matrix<T> result(lhs.getRows(), lhs.getColumns());

        for (size_t j = 0; j < lhs.getColumns(); ++j) {
            for (size_t i = 0; i < lhs.getRows(); ++i) {
                result[j][i] = lhs[j][i] * scalar;
            }
        }

        return result;
    }

    template <typename T>
    Matrix<T> operator*(const T& scalar, const Matrix<T>& lhs) {
        return operator*(lhs, scalar);
    }

    template <typename T>
    Matrix<T> operator/(const Matrix<T>& lhs, const T& scalar) {
        Matrix<T> result(lhs.getRows(), lhs.getColumns());

        for (size_t j = 0; j < lhs.getColumns(); ++j) {
            for (size_t i = 0; i < lhs.getRows(); ++i) {
                result[j][i] = lhs[j][i] / scalar;
            }
        }

        return result;
    }

    template <typename T>
    Matrix<T> operator/(const T& scalar, const Matrix<T>& lhs) {
        return operator/(lhs, scalar);
    }


    template <typename T>
    Matrix<T> operator*(const Matrix<T>& lhs, const Matrix<T>& rhs) {
        if (lhs.getColumns() != rhs.getRows()) {
            throw MatrixException("Matrix can't mult x1.cols != x2.rows matrices!");
        }

        Matrix<T> result(lhs.getRows(), rhs.getColumns());
        for (size_t j = 0; j < result.getColumns(); ++j) {
            for (size_t i = 0; i < result.getRows(); ++i) {
                for (size_t k = 0; k < lhs.getColumns(); ++k) {
                    result[j][i] += lhs[k][i] * rhs[j][k];
                }
            }
        }

        return result;
    }

    template <typename T>
    Matrix<T> operator/(const Matrix<T>& lhs, const Matrix<T>& rhs) {
        if (lhs.getColumns() != rhs.getRows()) {
            throw MatrixException("Matrix can't mult x1.cols != x2.rows matrices!");
        }

        Matrix<T> result(lhs.getRows(), rhs.getColumns());
        for (size_t j = 0; j < result.getColumns(); ++j) {
            for (size_t i = 0; i < result.getRows(); ++i) {
                for (size_t k = 0; k < lhs.getColumns(); ++k) {
                    result[j][i] += lhs[k][i] / rhs[j][k];
                }
            }
        }

        return result;
    }

    template <typename T>
    Matrix<T> transpose(const Matrix<T>& rhs) {
        Matrix<T> result(rhs.getColumns(), rhs.getRows());

        for (size_t j = 0; j < rhs.getColumns(); ++j) {
            for (size_t i = 0; i < rhs.getRows(); ++i) {
                result[i][j] = rhs[j][i];
            }
        }

        return result;
    }

    template <typename T>
    Matrix<T> minor(const Matrix<T>& rhs, size_t row_number, size_t col_number) {
        Matrix<T> result(rhs.getRows() - 1, rhs.getColumns() - 1);

        for (size_t j = 0, parent_j = 0; parent_j < rhs.getColumns(); ++parent_j) {
            if (parent_j == col_number) {
                continue;
            }

            for (size_t i = 0, parent_i = 0; parent_i < rhs.getRows(); ++parent_i) {
                if (parent_i == row_number) {
                    continue;
                }

                result[j][i] = rhs[parent_j][parent_i];
            }
            ++j;
        }

        return result;
    }

    template <typename T>
    T determinant(const Matrix<T>& rhs) {
        if (rhs.getColumns() != rhs.getRows()) {
            throw MatrixException("Matrix must have rows == columns for calc determinant!");
        }

        if (rhs.getColumns() == 2) {
            return ((rhs[0][0] * rhs[1][1]) - (rhs[0][1] * rhs[1][0]));
        }

        if (rhs.getColumns() == 1) {
            return rhs[0][0];
        }

        T result = 0;
        bool is_minus = false;
        for (size_t j = 0; j < rhs.getColumns(); ++j) {
            Matrix<T> minor_matrix = minor(rhs, j, 0);

            T column_result = rhs[j][0] * determinant(minor_matrix);
            result += (is_minus) ? -1 * column_result : column_result;

            is_minus = !is_minus;
        }

        return result;
    }

    template <typename T>
    Matrix<T> cofactor(const Matrix<T>& rhs) {
        Matrix<T> result(rhs.getRows(), rhs.getColumns());

        bool is_minus = false;
        bool is_minus_inner = false;
        for (size_t j = 0; j < rhs.getColumns(); ++j) {
            is_minus_inner = is_minus;
            for (size_t i = 0; i < rhs.getRows(); ++i) {
                Matrix<T> minor_matrix = minor(rhs, i, j);

                T determinant_result = determinant(minor_matrix);
                result[j][i] = (is_minus_inner) ? -1 * determinant_result: determinant_result;
                is_minus_inner = !is_minus_inner;
            }
            is_minus = !is_minus;
        }

        return result;
    }

    template <typename T>
    Matrix<T> inverse(const Matrix<T>& rhs) {
        return transpose(cofactor(rhs)) / determinant(rhs);
    }
}
}



#endif //ML_LINEAR_REGRESSION_MATRIX_OPERATION_H
