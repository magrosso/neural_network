// NeuralNetworks.cpp : This file contains the 'main' function. Program execution begins and ends there.//

#include <iostream>
#include <array>
#include "MLP.h"

using namespace std;

using LogicalGateInput = array<pair<double, double>, 4>;

void logicalGate(Perceptron &p, const vector<double> &weights, const LogicalGateInput &inputs, const string &operation);

void sdr_NN_7_to_10();

int main() {
    srand(time(nullptr));
    rand();

    cout << "\n\n--------Logic Gate Example----------------\n\n";

    // create a preceptron with two binary inputs + m_bias
    Perceptron p(2);

    cout << "AND Gate: " << endl;
    // set weights to simulate a logical AND gate

    const LogicalGateInput binaryInput{make_pair(0.0, 0.0),
                                       make_pair(0.0, 1.0),
                                       make_pair(1.0, 0.0),
                                       make_pair(1.0, 1.0)};

    using namespace string_literals;

    const vector<double> andWeights{10, 10, -15};
    logicalGate(p, andWeights, binaryInput, "AND"s);

    // set weights to simulate a logical OR gate
    const vector<double> orWeights{20, 20, -10};
    logicalGate(p, orWeights, binaryInput, "OR"s);

    const vector<double> nandWeights{-15, -15, 20};
    logicalGate(p, nandWeights, binaryInput, "NAND"s);

    MultiLayerPerceptron mp_xor({2, 2, 1});
    // XOR: (A NAND B) AND (A OR B)
    // Layer 1: Z1 = A NAND B, Z2 = A OR B
    // Layer 2: Z3 = Z1 AND Z2
    const vector<vector<double>> layer1{nandWeights, orWeights};
    const vector<vector<double>> layer2{andWeights};
    const vector<vector<vector<double>>> weights{layer1, layer2};

    mp_xor.set_weights(weights);
    mp_xor.print_weights();

    for (const auto input : binaryInput)
        cout << "A=" << input.first << " " << "XOR" << " B=" << input.second << " ==> "
             << mp_xor.run({input.first, input.second}).at(0)
             << "\n";

    //test code - Trained XOR
    cout << "\n\n--------Trained XOR Example----------------\n\n";
    auto trained_mlp = MultiLayerPerceptron({2, 2, 1});
    cout << "Training Neural Network as an XOR Gate...\n";
    // run the training set
    for (auto epoch{0}; epoch < 3000; ++epoch) {
        auto mse{0.0};
        mse += trained_mlp.backPropagation({0, 0}, {0});
        mse += trained_mlp.backPropagation({0, 1}, {1});
        mse += trained_mlp.backPropagation({1, 0}, {1});
        mse += trained_mlp.backPropagation({1, 1}, {0});
        mse /= static_cast<double>(4.0);
        if (epoch % 100 == 0)
            cout << "mse = " << mse << endl;
    }

    cout << "\n\nTrained weights (Compare to hard-coded weights):\n\n";
    trained_mlp.print_weights();

    cout << "XOR:" << endl;
    cout << "0 0 = " << trained_mlp.run({0, 0})[0] << endl;
    cout << "0 1 = " << trained_mlp.run({0, 1})[0] << endl;
    cout << "1 0 = " << trained_mlp.run({1, 0})[0] << endl;
    cout << "1 1 = " << trained_mlp.run({1, 1})[0] << endl;

    sdr_NN_7_to_10();
}


void sdr_NN_7_to_10() {
    //test code - Segment Display Recognition System
    cout << "Training Neural Network as an 7-segment digit recognition...\n";

    // create a multi layer perceptron with
    // input layer: 7 inputs
    // one hidden layer: 7 inputs
    // output layer: 10 outputs (one-hot encoding)
    auto sdrnn = MultiLayerPerceptron({7, 7, 10});

    const vector<vector<double>> segment_pattern{
            {1, 1, 1, 1, 1, 1, 0},   // digit 0 pattern
            {0, 1, 1, 0, 0, 0, 0},   // digit 1 pattern
            {1, 1, 0, 1, 1, 0, 1},   // digit 2 pattern
            {1, 1, 1, 1, 0, 0, 1},  // digit 3 pattern
            {0, 1, 1, 0, 0, 1, 1},   // digit 4 pattern
            {1, 0, 1, 1, 0, 1, 1},   // digit 5 pattern
            {1, 0, 1, 1, 1, 1, 1},   // digit 6 pattern
            {1, 1, 1, 0, 0, 0, 0},  // digit 7 pattern
            {1, 1, 1, 1, 1, 1, 1},  // digit 8 pattern
            {1, 1, 1, 1, 0, 1, 1}   // digit 9 pattern
    };

    // expected outputs
    const vector<vector<double>> segment_output{
            {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
    };

    // Dataset for the 7 to 1 network
    for (auto epoch{0}; epoch < 1000; ++epoch) {
        auto mse{0.0};
        for (const auto value: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
            mse += sdrnn.backPropagation(segment_pattern[value], segment_output[value]);
        mse /= 10.0;
        if (epoch % 100 == 0)
            cout << "MSE: " << mse << endl;
    }

    //sdrnn.print_weights();

    // test trained network
    for (const auto val: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
        cout << to_string(val) << " = " << sdrnn.run(segment_pattern[val])[val] << "\n";
}

void
logicalGate(Perceptron &p, const vector<double> &weights, const LogicalGateInput &inputs, const string &operation) {

    p.set_weights(weights);

    for (const auto input : inputs)
        cout << "A=" << input.first << " " << operation << " B=" << input.second << " ==> "
             << p.run({input.first, input.second})
             << "\n";
}

