#pragma once
#include <vector>
using matrix = std::vector<std::vector<float>>;

matrix tanh(matrix A);
matrix d_tanh(matrix A);
matrix sigmoid(matrix A);
matrix d_sigmoid(matrix A);
matrix relu(matrix A);
matrix d_relu(matrix A);