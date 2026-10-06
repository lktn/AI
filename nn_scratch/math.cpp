#include <iostream>
#include "math.hpp"
#include <iomanip>
#include <windows.h>

void print_vector(const std::vector<float> & a)
{
    std::cout << std::fixed << std::setprecision(6);
    SetConsoleOutputCP(CP_UTF8);

    std::cout << "[";
    for (size_t i = 0; i < a.size() - 1; i++) {
        if (a[i] < 0) {
            std::cout << a[i] << ", ";
        } else {
            std::cout << " " << a[i] << ", ";
        }
    }
    float x = a[a.size() - 1];
    if (x < 0) {
        std::cout << x << "]\n";
    } else {
        std::cout << " " << x << "]\n";
    };
}

void print_matrix(const matrix& A)
{
    SetConsoleOutputCP(CP_UTF8);

    std::cout << std::fixed << std::setprecision(6);

    // Độ rộng phần nội dung
    size_t width = A[0].size() * 9 + (A[0].size() - 1) * 2;

    // Khung trên
    std::cout << "┌" << std::string(width + 2, ' ') << "┐\n";

    // Ma trận
    for (size_t i = 0; i < A.size(); i++)
    {
        std::cout << "│ ";

        for (size_t j = 0; j < A[i].size(); j++) {
            if (j > 0)
                std::cout << "  ";
            std::cout << std::setw(9) << A[i][j];
        }
        std::cout << " │\n";
    }

    std::cout << "└" << std::string(width + 2, ' ') << "┘\n";
}

matrix zeros(const int & rows, const int & cols)
{
    return matrix(rows, std::vector<float>(cols));
}

matrix transpose(const matrix & A)
{
    matrix B = zeros(A[0].size(), A.size());
    for (size_t i = 0; i < A[0].size(); i++) {
        for (size_t j = 0; j < A.size(); j++) {
            B[i][j] = A[j][i];
        }
    }
    return B;
}

matrix operator+(matrix A, const matrix & B)
{
    for (size_t i = 0; i < A.size(); i++) {
        for (size_t j = 0; j < A[0].size(); j++) {
            A[i][j] += B[i][j];
        }
    }
    return A;
}

matrix operator-(matrix A, const matrix & B)
{
    for (size_t i = 0; i < A.size(); i++) {
        for (size_t j = 0; j < A[0].size(); j++) {
            A[i][j] -= B[i][j];
        }
    }
    return A;
}

matrix operator*(matrix A, const matrix & B)
{
    for (size_t i = 0; i < A.size(); i++) {
        for (size_t j = 0; j < A[0].size(); j++) {
            A[i][j] *= B[i][j];
        }
    }
    return A;
}

float operator*(const std::vector<float> & A, const std::vector<float> & B)
{
    float x = 0.f;
    for (size_t i = 0; i < A.size(); i++) 
        x += A[i] * B[i];
    
    return x;
}

matrix operator*(float a, matrix B)
{
    for (size_t i = 0; i < B.size(); i++) {
        for (size_t j = 0; j < B[0].size(); j++) {
            B[i][j] *= a;
        }
    }
    return B;
}

matrix dot(const matrix& A, const matrix& B)
{
    matrix C(A.size(), std::vector<float>(B[0].size(), 0.f));

    for (size_t i = 0; i < A.size(); i++) {
        for (size_t j = 0; j < B[0].size(); j++) {
            for (size_t k = 0; k < A[0].size(); k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

float MSE(const matrix & A) {
    float loss = 0.f;
    for (float x: A[0]) loss += x * x;
    return loss / A[0].size();
}

#include <random>

matrix random_matrix(const int & rows, const int & cols)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    matrix W;
    for (size_t i = 0; i < rows; i++) {
        std::vector<float> a;
        for (size_t j = 0; j < cols; j++) {
            float weight = dist(gen);
            a.push_back(weight);
        }
        W.push_back(a);
    }
    return W;
}