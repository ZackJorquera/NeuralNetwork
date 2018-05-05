#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string>

#include <time.h>

#include "NeuralNet.h"

using namespace std;

void CreateDefault();
double dRand(double fMin, double fMax);

void loop();
vector<string> NextInput();
void PrintHelp();

void LoadNeuralNet(string loadFrom);
void CreateNewNeuralNet(vector<string> args);
void ForwardPropagate(vector<string> args);
void CalcCost(vector<string> args);
void SaveToFile(string fileLocation);
void BackPropagate(vector<string> args);

void trainAdding();

NeuralNet theNeuralNet({0});

int main(int argNums, char** argv)
{
	cout << "Zack's Neural Network." << endl;
	cout << "A basic neural network." << endl;
	cout << "Created by Zack Jorquera." << endl;

	if (argNums > 1)
		LoadNeuralNet(argv[1]);

	srand(time(NULL));

	loop();

	return 0;
}

void loop()
{
	vector<string> lastInput = { "" };

	while (true)
	{
		vector<string> input = NextInput();
		if (input[0] == "")
			input = lastInput;

		if (input[0] == "quit")
			return;
		else if (input[0] == "fp")
			ForwardPropagate(input);
		else if (input[0] == "bp")
			BackPropagate(input);
		else if (input[0] == "cost")
			CalcCost(input);
		else if (input[0] == "load")
			LoadNeuralNet(input[1]);
		else if (input[0] == "create")
			CreateNewNeuralNet(input);
		else if (input[0] == "help")
			PrintHelp();
		else if (input[0] == "save")
			SaveToFile(input[1]);
		else if (input[0] == "t")
			trainAdding();
		else
			cout << "Undefined command: \"" << input[0] << "\".  Try \"help\"."<< endl;

		lastInput = input;
	}
}

void LoadNeuralNet(string loadFrom)
{
	if (loadFrom == "default")
	{
		CreateDefault();
		cout << "Succesfully loaded default neural net." << endl;
	}
	else
	{
		string zsonData = "";
		ifstream myFile;
		myFile.open(loadFrom);
		string line;
		while (getline(myFile, line))
		{
			zsonData += line + "\n";
		}
		myFile.close();
		string retString = "";
		theNeuralNet = NeuralNet(zsonData, &retString);
		if(retString != "")
			cout << "Failed to load neural net from file " << loadFrom << ".\n" << retString << endl;
		else
			cout << "Succesfully loaded neural net from " << loadFrom << "." << endl;
	}
}

void SaveToFile(string fileLocation)
{
	ofstream myfile;
	myfile.open(fileLocation);
	myfile << theNeuralNet.ExportNetworkToZSON();
	myfile.close();
}

void CreateNewNeuralNet(vector<string> args)
{
	int i = 1;
	string actvFunc;
	if (args[i] == "-a")
	{
		i += 2;
		actvFunc = args[i - 1];
	}

	vector<int> nndims;
	for (i = i; i < args.size(); i++)
	{
		nndims.push_back(stoi(args[i]));
	}
	
	theNeuralNet = NeuralNet(nndims);
	if (actvFunc != "")
	{
		if (!theNeuralNet.TrySetActivationFunctionType(actvFunc))
			cout << "Failed to set activation function type to \"" << actvFunc << "\", set to ReLU." << endl;//RelU is Default
	}
}

vector<string> NextInput()
{
	string input;
	cout << "(znn) ";
	getline(cin, input);

	vector<std::string> argArray;
	size_t pos = 0, found;
	while ((found = input.find_first_of(' ', pos)) != string::npos) {
		argArray.push_back(input.substr(pos, found - pos));
		pos = found + 1;
	}
	argArray.push_back(input.substr(pos));

	return argArray;
}

void CreateDefault()
{
	theNeuralNet = NeuralNet(vector<int>{ 3, 4, 2 });

	*theNeuralNet.weightAt(1, 0, 0) = 0.05;
	*theNeuralNet.weightAt(1, 1, 0) = 0.1;
	*theNeuralNet.weightAt(1, 2, 0) = 0.15;
	*theNeuralNet.weightAt(1, 3, 0) = 0.2;
	
	*theNeuralNet.weightAt(1, 0, 1) = 0.25;
	*theNeuralNet.weightAt(1, 1, 1) = 0.3;
	*theNeuralNet.weightAt(1, 2, 1) = 0.35;
	*theNeuralNet.weightAt(1, 3, 1) = 0.4;
	
	*theNeuralNet.weightAt(1, 0, 2) = -0.45;
	*theNeuralNet.weightAt(1, 1, 2) = 0.5;
	*theNeuralNet.weightAt(1, 2, 2) = 0.55;
	*theNeuralNet.weightAt(1, 3, 2) = 0.6;
	
	
	*theNeuralNet.weightAt(2, 0, 0) = 0.1;
	*theNeuralNet.weightAt(2, 1, 0) = -0.2;
	
	*theNeuralNet.weightAt(2, 0, 1) = 1;
	*theNeuralNet.weightAt(2, 1, 1) = 0.4;
	
	*theNeuralNet.weightAt(2, 0, 2) = 0.5;
	*theNeuralNet.weightAt(2, 1, 2) = 0.6;
	
	*theNeuralNet.weightAt(2, 0, 3) = 0.7;
	*theNeuralNet.weightAt(2, 1, 3) = 0.8;
	
	
	*theNeuralNet.biasAt(1, 0) = -0.3;
	*theNeuralNet.biasAt(1, 1) = -0.1;
	*theNeuralNet.biasAt(1, 2) = 0.25;
	*theNeuralNet.biasAt(1, 3) = -0.4;
	
	*theNeuralNet.biasAt(2, 0) = -0.1;
	*theNeuralNet.biasAt(2, 1) = 0.2;
}

void ForwardPropagate(vector<string> args)
{
	if (args.size() <= 1)
	{
		cout << "No args where given." << endl;
		return;
	}

	if (theNeuralNet.InputSize == 0)
	{
		cout << "The neural network has not been set up." << endl;
		return;
	}

	if (args[1] != "-r" && args.size() - 1 != theNeuralNet.InputSize)
	{
		cout << "Incorrect amount of arguments. Expected " << theNeuralNet.InputSize << ", but got " << args.size() - 1 << "." << endl;
		return;
	}

	vector<double> input;
	for (int i = 0; i < theNeuralNet.InputSize; i++)
	{
		if(args[1] == "-r")
			input.push_back(dRand(0, 1));
		else
			input.push_back(stod(args[i + 1]));
	}

	vector<double> output = theNeuralNet.ForwardPropagateAndCap(input);

	cout << "input:" << endl;
	for (int i = 0; i < input.size(); i++)
	{
		cout << "\t" << input[i] << endl;
	}

	cout << "output:" << endl;
	for (int i = 0; i < output.size(); i++)
	{
		cout << "\t" << output[i] << endl;
	}
}

void CalcCost(vector<string> args)
{
	if (args.size() - 1 != theNeuralNet.OutputSize)
	{
		cout << "Incorrect amount of arguments. Expected " << theNeuralNet.OutputSize << ", but got " << args.size() - 1 << "." << endl;
		return;
	}

	vector<double> target;
	for (int i = 0; i < theNeuralNet.OutputSize; i++)
	{
		target.push_back(stod(args[i + 1]));
	}

	cout << "cost :" << endl;
	cout << "\t" << theNeuralNet.CalculateCostFromOutput(target) << endl;
}

double dRand(double fMin, double fMax)
{
	double f = (double)(rand()) / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

void PrintHelp()
{
	cout << "List of commands:" << endl << endl << endl;

	cout << "fp [Params IntputVector/Options] - Does forward propagation with input vector." << endl;
	cout << "                          -r     - Uses a random input vector." << endl << endl;
	cout << "bp [Params TargetVector] - Performs a backpropagates using the previous forward propagation. Always use \"fp\" before to be safe." << endl << endl;
	cout << "cost [Params TargetVector] - Finds the cost with a given target vector. Should be used after a forward propagation" << endl << endl;
	cout << "load [File]  - Loads a Neural Network from a save file." << endl;
	cout << "     default - Loads the default test Neural Network." << endl << endl;
	cout << "save [File]  - Saves the current Neural Network to a save file." << endl << endl;
	cout << "create [-a [type]] [Params Dimensions] - Creates a new randomly filled Neural Network with the given dimensions." << endl << endl;

	//TODO: add help
}

void BackPropagate(vector<string> args)
{
	if (args.size() <= 1)
	{
		cout << "No args where given." << endl;
		return;
	}

	if (theNeuralNet.OutputSize == 0)
	{
		cout << "The neural network has not been set up." << endl;
		return;
	}

	if (args[1] != "-r" && args.size() - 1 != theNeuralNet.OutputSize)
	{
		cout << "Incorrect amount of arguments. Expected " << theNeuralNet.OutputSize << ", but got " << args.size() - 1 << "." << endl;
		return;
	}

	vector<double> targetVector;
	for (int i = 0; i < theNeuralNet.OutputSize; i++)
	{
		if (args[1] == "-r")
			targetVector.push_back(dRand(0, 1));
		else
			targetVector.push_back(stod(args[i + 1]));
	}

	vector<vector<double>> inputVectors;
	vector<vector<double>> targetVectors;
	targetVectors.push_back(targetVector);

	double change = theNeuralNet.BackPropagate(inputVectors, targetVectors, true, false);

	cout << "Change in Cost:\n\t" << to_string(change) << endl;
}

void trainAdding()
{
	for (int loop = 0; loop < 100; loop++)
	{
		vector<vector<double>> inputVectors;
		vector<vector<double>> targetVectors;
		for (int i = 0; i < 100; i++)
		{
			double sum = dRand(0, 1);
			double a = dRand(-1, 1);
			double b = sum - a;
			//double m = a * b;
			inputVectors.push_back(vector<double>{a, b});
			targetVectors.push_back(vector<double>{sum});
		}

		double change = theNeuralNet.BackPropagate(inputVectors, targetVectors, true, true);

		cout << "Change in Cost:\n\t" << to_string(change) << endl;
	}
}
