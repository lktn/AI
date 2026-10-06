#pragma once
#include <vector>
using matrix = std::vector<std::vector<float>>;

void print_matrix(const matrix & A);
void print_vector(const std::vector<float> & a);

matrix transpose(const matrix & A);
matrix operator+(matrix A, const matrix & B);
matrix operator-(matrix A, const matrix & B);
matrix operator*(matrix A, const matrix & B);
matrix operator*(float a, matrix B);
float operator*(const std::vector<float> & A, const std::vector<float> & B);
matrix dot(const matrix & A, const matrix & B);
matrix random_matrix(const int & rows, const int & cols);
matrix zeros(const int & rows, const int & cols);
float MSE(const matrix & A);