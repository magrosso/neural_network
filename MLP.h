#pragma once

#include <algorithm>
#include <vector>
#include <iostream>
#include <random>
#include <numeric>
#include <cmath>
#include <time.h>

using namespace std;

class Perceptron {
public:
    vector<double> weights;
    double bias;

    Perceptron(int inputCount, double bias = 1.0);

    double run(vector<double> input);

    void set_weights(vector<double> w_init);

    double sigmoid(double x);
};

class MultiLayerPerceptron {
public:
    explicit MultiLayerPerceptron(const vector<int>& layers, double bias = 1.0, double eta = 0.5);

    void set_weights(const vector<vector<vector<double>>>& w_init);

    void print_weights();

    vector<double> run(const vector<double> &input);

    double backPropagation(const vector<double> &out, const vector<double> &y);

    vector<int> m_layers;     // number of neurons per layer including number of inputs for input layer
    double m_bias;// input m_bias
    double m_eta; // learning rate
    vector<vector<Perceptron> > m_network;  // 2D matrix of Perceptrons: layer x neurons
    vector<vector<double>> m_values; // output m_values of neurons
    vector<vector<double>> m_deltas;  // error terms of the neurons
};

