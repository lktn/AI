#include <iostream>
#include <vector>
#include "math.hpp"
#include "activation_function.hpp"
//g++ nn_scratch.cpp math.cpp activation_function.cpp -o main && .\main
class MyModel {
private:
    matrix W1 = random_matrix(10, 2);
    matrix B1 = zeros(10, 1);

    matrix W2 = random_matrix(10, 10);
    matrix B2 = zeros(10, 1);

    matrix W3 = random_matrix(1, 10);
    matrix B3 = zeros(1, 1);

    matrix Z1, A1, Z2, A2, Z3, A3;

public:
    matrix forward(matrix input)
    {
        this-> Z1 = dot(W1, input) + B1;
        this-> A1 = tanh(Z1);

        this-> Z2 = dot(W2, A1) + B2;
        this-> A2 = tanh(Z2);

        this-> Z3 = dot(W3, A2) + B3;
        return sigmoid(Z3);
    }
    void backpropagation(
        matrix x_train, 
        matrix y_train, 
        size_t epochs, 
        float learning_rate
    )
    {
        for (size_t epoch = 1; epoch < epochs + 1; epoch++) {
            float total_loss = 0.f;
            for (size_t i = 0; i < x_train.size(); i++) {
                matrix x = transpose({x_train[i]});
                matrix y = transpose({y_train[i]});
                A3 = forward(x);

                matrix dL_dA3 = A3 - y;
                total_loss += MSE(dL_dA3);

                matrix dL_dZ3 = dL_dA3 * d_sigmoid(A3);
                matrix dL_dW3 = dot(dL_dZ3, transpose(A2));

                matrix dL_dA2 = dot(transpose(W3), dL_dZ3);

                matrix dL_dZ2 = dL_dA2 * d_tanh(A2);
                matrix dL_dW2 = dot(dL_dZ2, transpose(A1));

                matrix dL_A1 = dot(transpose(W2), dL_dZ2);

                matrix dL_dZ1 = dL_A1 * d_tanh(A1);
                matrix dL_dW1 = dot(dL_dZ1, transpose(x));

                W3 = W3 - learning_rate * dL_dW3;
                B3 = B3 - learning_rate * dL_dZ3;
                W2 = W2 - learning_rate * dL_dW2;
                B2 = B2 - learning_rate * dL_dZ2;
                W1 = W1 - learning_rate * dL_dW1;
                B1 = B1 - learning_rate * dL_dZ1;
            }
            if (epoch % 500 == 0)
            std::cout << "Epoch: " << epoch << ", loss: " << total_loss << "\n";
        }
    }
};
int main() {
    MyModel model;
    matrix x_train = {
        {0.f, 0.f},
        {0.f, 1.f},
        {1.f, 0.f},
        {1.f, 1.f}
    };

    matrix y_train = {
        {0.f},
        {1.f},
        {1.f},
        {0.f}
    };
    //matrix output = model.forward(x);

    model.backpropagation(x_train, y_train, 100000, 0.2f);

    return 0;
}