#include <cmath>
#include <vector>
#include "activation_function.hpp"

using matrix = std::vector<std::vector<float>>;

matrix tanh(matrix A)
{
    for (size_t i = 0; i < A.size(); i++) {
        for (size_t j = 0; j < A[0].size(); j++){
            A[i][j] = tanh(A[i][j]);
        }
    }
    return A;
}
matrix d_tanh(matrix A)
{
    for (size_t i = 0; i < A.size(); i++) {
        for (size_t j = 0; j < A[0].size(); j++){
            A[i][j] = 1 - A[i][j] * A[i][j];
        }
    }
    return A;
}

matrix sigmoid(matrix A)
{
    for (size_t i = 0; i < A.size(); i++) {
        for (size_t j = 0; j < A[0].size(); j++){
            A[i][j] = 1 / (1 + exp(-A[i][j]));
        }
    }
    return A;
}
matrix d_sigmoid(matrix A)
{
    for (size_t i = 0; i < A.size(); i++) {
        for (size_t j = 0; j < A[0].size(); j++){
            A[i][j] = A[i][j] * (1 - A[i][j]);
        }
    }
    return A;
}
matrix relu(matrix A)
{
    for (size_t i = 0; i < A.size(); i++) {
        for (size_t j = 0; j < A[0].size(); j++){
            if (A[i][j] < 0)
                A[i][j] = 0.f;
        }
    }
    return A;
}
matrix d_relu(matrix A)
{
    for (size_t i = 0; i < A.size(); i++) {
        for (size_t j = 0; j < A[0].size(); j++){
            if (A[i][j] > 1)
                A[i][j] = 1.f;
            else
                A[i][j] = 0.f;
        }
    }
    return A;
}