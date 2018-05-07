#include "NeuralNet.h"
#include <math.h>
#include <iomanip>
#include <sstream>

#define e 2.71828182845904523536028747135266249775724709369995   //Might be to much
#define loge 0.43429448190325182765112891891660508229439700580366
#define ln(v) log(v)/loge

namespace hp
{
	std::string to_string(double d)
	{
		std::ostringstream stm;
		stm << std::setprecision(std::numeric_limits<double>::max_digits10) << d;
		return stm.str();
	}
}

using namespace std;


double pdRand(double fMin, double fMax);
string grabChunk(string, int *startVal);
vector<double> GetCSDoubles(string csvs);

vector<int> GetDimensions(string generalInfo);
string GetActivationFunctionType(string generalInfo);
vector<vector<vector<double>>> GetWeights(string weightsInfo);
vector<vector<double>> GetBiases(string biasesInfo);
string ToUpper(string);


NeuralNet::NeuralNet(vector<int> dimensions)
{
	neurons = vector<std::vector<double>>();
	weights = vector<std::vector<std::vector<double>>>();
	biases = vector<std::vector<double>>();

	InputSize = dimensions[0];
	OutputSize = dimensions[dimensions.size() - 1];
	for (unsigned i = 0; i < dimensions.size(); i++)
	{
		vector<double> neuronLayer;
		for (int j = 0; j < dimensions[i]; j++)
			neuronLayer.push_back(0);
		neurons.push_back(neuronLayer);

		if (i != 0)
		{
			vector<double> biasLayer;
			for (int j = 0; j < dimensions[i]; j++)
				biasLayer.push_back(0);//(pdRand(-1, 1));
			biases.push_back(biasLayer);

			vector<vector<double>> layerWeights;
			for (int to = 0; to < dimensions[i]; to++)
			{
				vector<double> weightLayer;
				for (int from = 0; from < dimensions[i - 1]; from++)
					weightLayer.push_back(pdRand(-1, 1));
				layerWeights.push_back(weightLayer);
			}
			weights.push_back(layerWeights);
		}
	}
}

NeuralNet::NeuralNet(string zsonData, string *retString)
{//TODO: use a try catch also make is use less memory to open because it uses about 8 times the size of the file and takes like 10000000 years
	string generalInfo = "";
	string activationFunctionType = "";
	string weightsInfo = "";
	string biasesInfo = "";

	for (int i = 0; i < zsonData.length(); i++)
	{
		if (zsonData[i] == '{')
		{
			string chunk = grabChunk(zsonData, &i);
			if (chunk[1] == 'd' && chunk[2] == 'i')
				generalInfo = chunk;
			else if (chunk[1] == 'w' && chunk[2] == 'e')
				weightsInfo = chunk;
			else if (chunk[1] == 'b' && chunk[2] == 'i')
				biasesInfo = chunk;
		}
	}
	if (generalInfo == "" || weightsInfo == "" || biasesInfo == "")
	{
		*retString = "Bad save file.";
		neurons = vector<vector<double>>();
		weights = vector<vector<vector<double>>>();
		biases = vector<vector<double>>();
	}
	else
	{
		neurons = vector<vector<double>>();
		weights = vector<vector<vector<double>>>();
		biases = vector<vector<double>>();

		vector<int> dimensions = GetDimensions(generalInfo);
		activationFunctionType = GetActivationFunctionType(generalInfo);
		TrySetActivationFunctionType(activationFunctionType);

		InputSize = dimensions[0];
		OutputSize = dimensions[dimensions.size() - 1];
		for (unsigned i = 0; i < dimensions.size(); i++)
		{
			vector<double> neuronLayer;
			for (int j = 0; j < dimensions[i]; j++)
				neuronLayer.push_back(0);
			neurons.push_back(neuronLayer);
		}

		weights = GetWeights(weightsInfo);
		biases = GetBiases(biasesInfo);

		if (neurons.size() == 0 || weights.size() == 0 || biases.size() == 0)
		{
			*retString = "Bad save file.";
			neurons = vector<vector<double>>();
			weights = vector<vector<vector<double>>>();
			biases = vector<vector<double>>();
		}
	}
}

vector<int> GetDimensions(string generalInfo)
{
	vector<int> dims;
	for (int i = 0; i < generalInfo.length(); i++)
	{
		if (generalInfo[i] == '[')
		{
			vector<double> doubleDims = GetCSDoubles(grabChunk(generalInfo, &i));
			for (int i = 0; i < doubleDims.size(); i++)
				dims.push_back(int(doubleDims[i]));
			break;
		}
	}
	return dims;
}

string GetActivationFunctionType(string generalInfo)
{
	string actFunc = "";
	for (int i = 0; i < generalInfo.length(); i++)
	{
		if (generalInfo[i] == ':' && generalInfo[i - 4] == 'f' && generalInfo[i - 7] == 'a')
		{
			for (int j = 1; generalInfo[i + j] != '}' && generalInfo[i + j] != ','; j++)
			{
				actFunc += generalInfo[i + j];
			}
			return actFunc;
		}
	}
}

vector<vector<vector<double>>> GetWeights(string weightsInfo)
{
	vector<vector<vector<double>>> theWeights;

	for (int i = 0; i < weightsInfo.length(); i++)
	{
		if (weightsInfo[i] == '[')
		{
			vector<vector<double>> layerWeights;
			string chunk = grabChunk(weightsInfo, &i);
			chunk = chunk.substr(1, chunk.size() - 2);
			for (int j = 0; j < chunk.length(); j++)
			{
				if(chunk[j] == '[')
					layerWeights.push_back(GetCSDoubles(grabChunk(chunk, &j)));
			}
			theWeights.push_back(layerWeights);
		}
	}

	return theWeights;
}

vector<vector<double>> GetBiases(string biasesInfo)
{
	vector<vector<double>> theBiases;
	
	for (int i = 0; i < biasesInfo.length(); i++)
	{
		if (biasesInfo[i] == '[')
		{
			theBiases.push_back(GetCSDoubles(grabChunk(biasesInfo, &i)));
		}
	}

	return theBiases;
}

vector<double> GetCSDoubles(string csvs)
{
	csvs = csvs.substr(1, csvs.size() - 2);

	vector<double> valArray;
	vector<string> argArray;
	size_t pos = 0, found;
	while ((found = csvs.find_first_of(',', pos)) != string::npos) {
		argArray.push_back(csvs.substr(pos, found - pos));
		pos = found + 1;
	}
	argArray.push_back(csvs.substr(pos));

	for (int i = 0; i < argArray.size(); i++)
	{
		double val = 1;
		if (argArray[i][0] == '-')
		{
			val = -1;
			argArray[i] = argArray[i].substr(1, argArray[i].size() - 1);
		}
		val *= stod(argArray[i]);
		valArray.push_back(val);
	}

	return valArray;
}

string grabChunk(string str, int *val)
{
	char startChar = str[*val];
	char endChar;
	int endCharsLeftBeforeEnd = 0;
	if (startChar == '{')
		endChar = '}';
	else if (startChar == '[')
		endChar = ']';
	else if (startChar == '(')
		endChar = ')';
	else
		return str;

	string returnString = "";
	for (*val = *val; *val < str.size(); (*val)++)
	{
		if (str[*val] == ' ' || str[*val] == '\n' || str[*val] == '\t')
			continue;
		returnString += str[*val];
		if (str[*val] == startChar)
			endCharsLeftBeforeEnd++;
		else if (str[*val] == endChar)
		{
			endCharsLeftBeforeEnd--;
			if(endCharsLeftBeforeEnd <= 0)
				break;
		}
	}
	return returnString;
}

vector<double> NeuralNet::ForwardPropagate(vector<double> inputVector)
{
	if (InputSize == inputVector.size())
	{
		for (unsigned i = 0; i < inputVector.size(); i++)
		{
			neurons[0][i] = inputVector[i];
		}

		for (unsigned i = 0; i < neurons.size() - 1; i++)
		{
			ForwardPropagateOneLayer(i);
		}
		return neurons[neurons.size() - 1];
	}
	else
		throw "Bad arguments.";
	return vector<double>();
}

vector<double> NeuralNet::ForwardPropagateAndCap(vector<double> inputVector)
{
	vector<double> output = ForwardPropagate(inputVector);

	for (unsigned i = 0; i < output.size(); i++)
	{
		if (output[i] > 1)
			output[i] = 1;
		if (output[i] < 0)
			output[i] = 0;
	}
	
	return(output);
}

void NeuralNet::ForwardPropagateOneLayer(int layer)
{
	for (unsigned to = 0; to < neurons[layer + 1].size(); to++)
	{
		double newValue = biases[(layer + 1) - 1][to];
		for (unsigned from = 0; from < neurons[layer].size(); from++)
		{
			newValue += weights[layer][to][from] * neurons[layer][from];
		}
		neurons[layer + 1][to] = ActivationFunction(newValue, false);
	}
}

double NeuralNet::ActivationFunction(double v, bool derivative)
{
	if (_activationFunctionTypes[_activationFunctionType] == "PRELU")
	{
		double a = 0.001;
		if (derivative)
			if (v <= 0)
				v = a;
			else
				v = 1;
		else
			if (v <= 0)
				v = v * a;
			else
				v = v;
	}
	else if (_activationFunctionTypes[_activationFunctionType] == "TANH")
	{
		if (derivative)
			v = 1 - pow(tanh(v), 2);
		else
			v = tanh(v);
	}
	else if (_activationFunctionTypes[_activationFunctionType] == "SOFTPLUS")
	{
		if (derivative)
			v = 1 / (1 + pow(e, v));
		else
			v = ln(1 + pow(e, v));//log(1 + pow(e, v)) / loge;
	}
	else if (_activationFunctionTypes[_activationFunctionType] == "LINEAR")
	{
		if (derivative)
			v = 1;
		else
			v = v;
	}
	return v;
}

bool NeuralNet::TrySetActivationFunctionType(string typeString)
{
	for (int i = 0; i < _activationFunctionTypes.size(); i++)
	{
		string a = _activationFunctionTypes[i];
		string b = ToUpper(typeString);
		if (a == b)
		{
			_activationFunctionType = i;
			return true;
		}
	}
	_activationFunctionType = 0;
	return false;
}

double NeuralNet::CalculateCostFromOutput(std::vector<double> targetVector)
{
	double cost = 0;
	if (neurons[neurons.size() -1].size() == targetVector.size())
	{
		for (unsigned i = 0; i < targetVector.size(); i++)
		{
			cost += pow(targetVector[i] - neurons[neurons.size() - 1][i], 2);
		}
		cost /= targetVector.size();
		return cost;
	}
	throw "Bad arguments.";
	return 1;
}

string NeuralNet::ExportNetworkToZSON()
{
	string output = "";

	output += "{\ndims:\n  [";
	for (int layer = 0; layer < neurons.size(); layer++)
	{
		output += to_string(neurons[layer].size());
		if (layer != neurons.size() - 1)
			output += ",";
	}
	output += "],\n";
	output += "actfunc:" + _activationFunctionTypes[_activationFunctionType] + "\n}\n";

	output += "{\nweights:";//TODO: add [] around the weights values
	for (int layer = 0; layer < weights.size(); layer++)
	{
		output += "\n  [";
		for (int to = 0; to < weights[layer].size(); to++)
		{
			output += "\n    [";
			for (int from = 0; from < weights[layer][to].size(); from++)
			{
				output += hp::to_string(weights[layer][to][from]);
				if (from != weights[layer][to].size() - 1)
					output += ",";
			}
			output += "]";
			if (to != weights[layer].size() - 1)
				output += ",";
		}
		output += "\n  ]";
		if (layer != weights.size() - 1)
			output += ",";
	}
	output += "\n}\n";

	output += "{\nbiases:";
	for (int layer = 0; layer < biases.size(); layer++)
	{
		output += "\n  [";
		for (int neuron = 0; neuron < biases[layer].size(); neuron++)
		{
			output += hp::to_string(biases[layer][neuron]);
			if (neuron != biases[layer].size() - 1)
				output += ",";
		}
		output += "]";
		if (layer != biases.size() - 1)
			output += ",";
	}
	output += "\n}\n";

	return output;
}

double NeuralNet::BackPropagate(std::vector<std::vector<double>> inputVectors, std::vector<std::vector<double>> targetVectors, double learningRate)
{
	vector<vector<vector<double>>> shiftWeights;
	vector<vector<double>> shiftBiases;

	if (inputVectors.size() != targetVectors.size() && inputVectors.size() != 0)
		throw "Arguments sizes do not equal.";

	for (int i = 0; i < targetVectors.size(); i++)
	{
		if (inputVectors.size() == 0)
			ForwardPropagate(neurons[0]);
		else
			ForwardPropagate(inputVectors[i]);

		vector<double> inputdC_das;
		for(int j = 0; j < neurons[neurons.size() - 1].size(); j++)
			inputdC_das.push_back(2 * (neurons[neurons.size() - 1][j] - targetVectors[i][j]));

		pair<vector<vector<double>>, vector<vector<vector<double>>>> retValue = BackPropagateAllLayers(inputdC_das, neurons.size() - 1);

		if (i == 0)//first time
		{
			shiftWeights = retValue.second;
			shiftBiases = retValue.first;
		}
		else
		{
			for (int l = 0; l < shiftWeights.size(); l++)
			{
				for (int j = 0; j < shiftWeights[l].size(); j++)
				{
					for (int k = 0; k < shiftWeights[l][j].size(); k++)
					{
						shiftWeights[l][j][k] = ((shiftWeights[l][j][k] * double(i)) + retValue.second[l][j][k]) / double(i + 1);
					}
					shiftBiases[l][j] = ((shiftBiases[l][j] * double(i)) + retValue.first[l][j]) / double(i + 1);
				}
			}
		}
	}
	for (int l = 0; l < weights.size(); l++)
	{
		for (int j = 0; j < weights[l].size(); j++)
		{
			for (int k = 0; k < weights[l][j].size(); k++)
			{
				weights[l][j][k] -= shiftWeights[l][j][k] * learningRate;
			}
			biases[l][j] -= shiftBiases[l][j];
		}
	}

	if (inputVectors.size() == 0)
		ForwardPropagate(neurons[0]);
	else
		ForwardPropagate(inputVectors[0]);

	double cost = CalculateCostFromOutput(targetVectors[0]);
	return cost;
}

pair<vector<vector<double>>, vector<vector<vector<double>>>> NeuralNet::BackPropagateAllLayers(vector<double> inputdC_das, int layerbpTo)
{
	if (layerbpTo == 0)
		return pair<vector<vector<double>>, vector<vector<vector<double>>>>();

	vector<double> dC_dbsForThisLayer;
	vector<vector<double>> dC_dwsForThisLayer;

	vector<double> outputdC_das;
	for (int j = 0; j < inputdC_das.size(); j++)
	{
		vector<double> dC_dbsForInputNeuron;


		double zval = biases[layerbpTo - 1][j];
		for (unsigned from = 0; from < neurons[layerbpTo - 1].size(); from++)
		{
			zval += weights[layerbpTo - 1][j][from] * neurons[layerbpTo - 1][from];
		}

		double da_dz = ActivationFunction(zval, true);
		
		for (int k = 0; k < neurons[layerbpTo - 1].size(); k++)
		{
			double thisdC_da = weights[layerbpTo - 1][j][k] * da_dz * inputdC_das[j];
			if (j == 0)//if first time
				outputdC_das.push_back(thisdC_da);
			else
				outputdC_das[k] += thisdC_da;

			double thisdz_dw = neurons[layerbpTo - 1][k];//weights[layerbpTo - 1][j][k];
			double thisdC_dw = thisdz_dw * da_dz * inputdC_das[j];
			dC_dbsForInputNeuron.push_back(thisdC_dw);
		}
		
		double dz_db = 0.0001;//biases[layerbpTo - 1][j];
		double dC_db = dz_db * da_dz * inputdC_das[j];

		dC_dbsForThisLayer.push_back(dC_db);
		dC_dwsForThisLayer.push_back(dC_dbsForInputNeuron);
	}
	
	pair<vector<vector<double>>, vector<vector<vector<double>>>> retValue = BackPropagateAllLayers(outputdC_das, layerbpTo - 1);
	
	retValue.first.push_back(dC_dbsForThisLayer);
	retValue.second.push_back(dC_dwsForThisLayer);

	return retValue;
}

double pdRand(double fMin, double fMax)
{
	double f = (double)(rand()) / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

string ToUpper(string str)
{
	for (string::iterator strIter = str.begin(); strIter != str.end(); ++strIter)
		*strIter = toupper(*strIter);
	return str;
}
