#ifndef NEURALNET_H
#define NEURALNET_H

#include <vector>
#include <string>

class NeuralNet
{
    int vtr;
    std::vector<std::vector<double>> neurons;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
public:
    int InputSize, OutputSize;
    bool TrySetActivationFunctionType(std::string);
    NeuralNet(std::vector<int> dimensions);
    NeuralNet(std::string zsonData, std::string *retString);
    std::vector<double> ForwardPropagate(std::vector<double> inputVector);
    std::vector<double> ForwardPropagateAndCap(std::vector<double> inputVector);
    double CalculateCostFromOutput(std::vector<double> targetVector);
    std::string ExportNetworkToZSON();
    double *weightAt(int layerTo, int to, int from) { return &(weights[layerTo - 1][to][from]); }
    double *biasAt(int layer, int neuron) { return &(biases[layer - 1][neuron]); }
    double *neuronAt(int layer, int neuron) { return &(neurons[layer][neuron]); }
    double BackPropagate(std::vector<std::vector<double>> inputVectors, std::vector<std::vector<double>> targetVectors, double learningRate);
private:
    double learningrate;
    std::vector<std::string> _activationFunctionTypes{ "PRELU", "TANH", "SOFTPLUS", "LINEAR", "SIGMOID"};
    int _activationFunctionType = 0;
    void ForwardPropagateOneLayer(int fromLayer);
    double ActivationFunction(double, bool);
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>> BackPropagateAllLayers(std::vector<double> dC_dz, int layer);
};

#endif