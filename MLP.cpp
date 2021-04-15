#include <cassert>
#include <algorithm>
#include "MLP.h"


double frand() {
    return 2.0 * (double) rand() / RAND_MAX - 1.0;
}


// Return a new Perceptron object with the specified number of inputCount (+1 for the m_bias).
Perceptron::Perceptron(int inputCount, double bias) : bias(bias) {
    weights.resize(inputCount + 1);
    std::generate(weights.begin(), weights.end(), frand);
}

// Run the perceptron. input is a vector with the input m_values.
double Perceptron::run(vector<double> input) {
    input.push_back(bias);
    auto sum = inner_product(input.begin(), input.end(), weights.begin(), 0.0);
    return sigmoid(sum);
}

void Perceptron::set_weights(vector<double> w_init) {
    assert(w_init.size() == weights.size());
    weights = w_init;
}

double Perceptron::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}


MultiLayerPerceptron::MultiLayerPerceptron(const vector<int> &layers, double bias, double eta) : m_layers(layers),
                                                                                                 m_bias(bias),
                                                                                                 m_eta(eta) {
    cout << "Initializing network with " << m_layers.size() << " layers" << "\n";

    // init all layers
    for (auto layer{0}; layer < m_layers.size(); ++layer) {
        // create neuronCount output vector and init to zero
        m_values.push_back(vector<double>(m_layers[layer], 0.0));
        m_deltas.push_back(vector<double>(m_layers[layer], 0.0));
        // init neuronCount with no neurons
        m_network.push_back(vector<Perceptron>());
        // init each neuron in this neuronCount
        if (layer > 0) {  // input layer has no neurons
            // for each neuron in neuronCount, create a Perceptron with as many inputs as the previous neuronCount has outputs
            for (auto neuron{0}; neuron < m_layers[layer]; ++neuron) {
                m_network[layer].push_back(Perceptron(m_layers[layer - 1], bias));
            }
        }
    }
}

void MultiLayerPerceptron::set_weights(const vector<vector<vector<double>>> &w_init) {
    assert(m_network.size() > w_init.size());

    for (auto layer{0}; layer < w_init.size(); ++layer) {
        for (auto neuron{0}; neuron < w_init[layer].size(); ++neuron) {
            m_network[layer + 1][neuron].set_weights(w_init[layer][neuron]);
        }
    }
}

void MultiLayerPerceptron::print_weights() {

    cout << "Printing weights with " << m_layers.size() << " layers" << "\n";

    for (auto layer{1}; layer < m_network.size(); ++layer) {
        for (auto neuron{0}; neuron < m_layers[layer]; ++neuron) {
            cout << "Weight: " << "layer=" << layer << ", neuron=" << neuron << ": ";
            for (auto weight:m_network[layer][neuron].weights)
                cout << weight << ",";
            cout << "\n";
        }
    }
}

vector<double> MultiLayerPerceptron::run(const vector<double> &input) {
    // copy input into first layer
    m_values.at(0) = input;

    // calc output values of each neuron in each layer starting with the 2nd layer
    for (auto layer{1}; layer < m_network.size(); ++layer) {
        for (auto neuron{0}; neuron < m_layers[layer]; ++neuron) {
            // run current neuron with input from previous layer
            m_values[layer][neuron] = m_network[layer][neuron].run(m_values[layer - 1]);
        }
    }
    return m_values.back();
}

// Run a single (x,y) pair with the backpropagation algorithm.
double MultiLayerPerceptron::backPropagation(const vector<double> &x, const vector<double> &y) {

    // Backpropagation Step by Step:
    // STEP 1: Feed a sample to the network
    const auto output = run(x);

    // STEP 2: Calculate the MSE
    // number of neurons in output layer and y must be same
    assert(y.size() == m_layers.back());
    auto squared_error_sum{0.0};
    for (auto neuron{0}; neuron < y.size(); ++neuron) {
        squared_error_sum += pow(y[neuron] - output[neuron], 2);
    }

    // mean squared error
    const auto mse = squared_error_sum / m_layers.back();

    // STEP 3: Calculate the output error terms (the error of each output layer neuron)
    // delta = output * (1 - output) * (y - output);
    for (auto neuron{0}; neuron < output.size(); ++neuron) {
        m_deltas.back()[neuron] = output[neuron] * (1 - output[neuron]) * (y[neuron] - output[neuron]);
    }

    // STEP 4: Calculate the error term of each unit on each layer
    // iterate from last hidden layer to first hidden layer
    const int hidden_layer_count{static_cast<int>(m_network.size()) - 2};
    assert(hidden_layer_count > 0);

    for (auto hidden_layer{hidden_layer_count}; hidden_layer > 0; --hidden_layer) {
        for (auto hidden_neuron{0}; hidden_neuron < m_network[hidden_layer].size(); ++hidden_neuron) {
            double fwd_error{0.0};
            // next layer after current hidden layer
            for (auto neuron{0}; neuron < m_layers[hidden_layer + 1]; ++neuron)
                fwd_error +=
                        m_deltas[hidden_layer + 1][neuron] * m_network[hidden_layer + 1][neuron].weights[hidden_neuron];
            // calc error term of current neuron
            m_deltas[hidden_layer][hidden_neuron] =
                    m_values[hidden_layer][hidden_neuron] * (1 - m_values[hidden_layer][hidden_neuron]) * fwd_error;
        }
    }

    // STEPS 5 & 6: Calculate the deltas and update the weights
    for (auto layer{1}; layer < m_network.size(); ++layer) {
        for (auto neuron{0}; neuron < m_layers[layer]; ++neuron) {
            // all inputs: number of neuron in previous layer +1 for the bias weight
            for (auto input{0}; input < m_layers[layer - 1] + 1; ++input) {
                // if input is bias weight
                const auto bias_weight = input == m_layers[layer - 1];
                const auto err{m_eta * m_deltas[layer][neuron]};
                const auto delta = bias_weight ? err * m_bias : err * m_values[layer - 1][input];
                // update weight
                m_network[layer][neuron].weights[input] += delta;
            }
        }
    }
    return mse;
}
