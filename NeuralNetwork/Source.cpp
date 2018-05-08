#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string>

#include <time.h>

#include <intrin.h>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

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

vector<double> LoadImage(string filePath);
void TrainHandWrittenNumsFromMNISTData(vector<string> input);
uint32_t readuint32FromFile(ifstream stream);

void trainAdding();

NeuralNet theNeuralNet({0});

int main(int argNums, char** argv)
{
	cout << "Zack's Neural Network." << endl;
	cout << "A basic neural network." << endl;
	cout << "Created by Zack Jorquera." << endl;

	if (argNums > 1)
		LoadNeuralNet(argv[1]);

	srand(uint(time(NULL)));

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
		else if (input[0] == "trainnums")
			TrainHandWrittenNumsFromMNISTData(input);//LoadImage(input[1]);//trainAdding();
		else if (input[0] == "trainsum")
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
			cout << "Failed to set activation function type to \"" << actvFunc << "\", set to PReLU." << endl;//RelU is Default
	}
}

vector<string> NextInput()
{
	string input;
	cout << "(znn) ";
	getline(cin, input);

	vector<std::string> argArray;
	size_t pos = 0;
	bool lookingForQuote = false;

	for (int i = 0; i < input.size(); i++)
	{
		if (lookingForQuote)
		{
			if (input[i] == '\"')
			{
				argArray.push_back(input.substr(pos, i - pos));
				lookingForQuote = false;
				pos = i + 1;
			}
		}
		else
		{
			if (input[i] == '\"')
			{
				lookingForQuote = true;
				pos = i + 1;
			}
			if (input[i] == ' ')
			{
				if(i - pos >= 1)
					argArray.push_back(input.substr(pos, i - pos));
				pos = i + 1;
			}
		}
	}
	if (pos != input.size())
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

	if ((args[1] != "-r" && args[1] != "-i") && args.size() - 1 != theNeuralNet.InputSize)
	{
		cout << "Incorrect amount of arguments. Expected " << theNeuralNet.InputSize << ", but got " << args.size() - 1 << "." << endl;
		return;
	}

	vector<double> input;
	if (args[1] != "-i")
	{
		for (int i = 0; i < theNeuralNet.InputSize; i++)
		{
			if (args[1] == "-r")
				input.push_back(dRand(0, 1));
			else
				input.push_back(stod(args[i + 1]));
		}
	}
	else
	{
		input = LoadImage(args[2]);
		if (input.size() != theNeuralNet.InputSize)
		{
			cout << "Incorrect image size. Expected " << theNeuralNet.InputSize << ", but got " << input.size() << "." << endl;
			return;
		}
	}

	vector<double> output = theNeuralNet.ForwardPropagate(input);//AndCap(input);

	cout << "input:" << endl;
	if (args[1] != "-i")
	{
		for (int i = 0; i < input.size(); i++)
		{
			cout << "\t" << input[i] << endl;
		}
	}
	else
		cout << "\tFrom image: " << args[2] << endl;

	int biggest = 0;
	for (int i = 1; i < output.size(); i++)
	{
		if (output[i] > output[biggest])
			biggest = i;
	}
	cout << "output:" << endl;
	for (int i = 0; i < output.size(); i++)
	{
		if (biggest == i)
			cout << "*";
		cout << i << ":" << "\t" << output[i] << endl;
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

	cout << "cost:" << endl;
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
	cout << "bp [Options] [Params TargetVector] - Performs a backpropagates using the previous forward propagation. Always use \"fp\" before to be safe." << endl;
	cout << "       -l                          - Set the learning rate, default is 0.5." << endl << endl;
	cout << "cost [Params TargetVector] - Finds the cost with a given target vector. Should be used after a forward propagation" << endl << endl;
	cout << "load [File]  - Loads a Neural Network from a save file." << endl;
	cout << "     default - Loads the default test Neural Network." << endl << endl;
	cout << "save [File]  - Saves the current Neural Network to a save file." << endl << endl;
	cout << "create [-a [type]] [Params Dimensions] - Creates a new randomly filled Neural Network with the given dimensions." << endl << endl;
	cout << "trainnums [Options] [Image File] [Label File] [Training Size] [Learning Rate] - Trains the network to detect handwritten numbers in 28 by 28 images using the MNIST database. Requires input vector size to be 784 and output size to be 10." << endl;
	cout << "             -p [Modular]                                                     - Print the cost as the network is being trained every time the iteration mod the modular is equal to zero." << endl;
	cout << "             -o [Offset]                                                      - Starts training at the offset." << endl;
	cout << "             -s [Step Size]                                                   - The step size when training the network." << endl << endl;

}

void BackPropagate(vector<string> args)
{
	double learningRate = 0.5;
	bool useRand = false;
	int argEndAt = 1;

	for (int iter = 1; iter < args.size(); iter++)
	{
		argEndAt = iter;
		if (args[iter] == "-l")
		{
			learningRate = stod(args[iter + 1]);
			iter++;
			argEndAt = iter;
		}
		else if (args[iter] == "-r")
			useRand = true;
		else
			break;
	}

	if (args.size() <= argEndAt && !useRand)
	{
		cout << "No args where given." << endl;
		return;
	}

	if (theNeuralNet.OutputSize == 0)
	{
		cout << "The neural network has not been set up." << endl;
		return;
	}

	if (!useRand && args.size() - argEndAt != theNeuralNet.OutputSize)
	{
		cout << "Incorrect amount of target vector arguments. Expected " << theNeuralNet.OutputSize << ", but got " << args.size() - 1 << "." << endl;
		return;
	}

	vector<double> targetVector;
	for (int i = 0; i < theNeuralNet.OutputSize; i++)
	{
		if (useRand)
			targetVector.push_back(dRand(0, 1));
		else
			targetVector.push_back(stod(args[i + argEndAt]));
	}

	vector<vector<double>> inputVectors;
	vector<vector<double>> targetVectors;
	targetVectors.push_back(targetVector);

	double cost = theNeuralNet.BackPropagate(inputVectors, targetVectors, learningRate);

	cout << "Cost:\n\t" << cost << endl;
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
			double rem = sum - a;
			double b = dRand(-1, 1);
			rem = rem - b;
			double c = dRand(-1, 1);
			double d = rem - c;
			
			//double m = a * b;
			inputVectors.push_back(vector<double>{a, b, c, d});
			targetVectors.push_back(vector<double>{sum});
		}

		double cost = theNeuralNet.BackPropagate(inputVectors, targetVectors, 0.5);

		cout << "Cost:\n\t" << cost << endl;
	}
}

void TrainHandWrittenNumsFromMNISTData(vector<string> input)
{
	int startAt = 0;
	string dataFileString;
	string labelFileString;
	int trainingSize;
	int stepSize = 1;
	int printModular = 0;
	double learningRate;

	for (int iter = 1; iter < input.size(); iter++)
	{
		if (input[iter] == "-p")
		{
			printModular = stoi(input[iter + 1]);
			iter++;
		}
		else if (input[iter] == "-o")
		{
			startAt = stoi(input[iter + 1]);
			iter++;
		}
		else if (input[iter] == "-s")
		{
			stepSize = stoi(input[iter + 1]);
			iter++;
		}
		else
		{
			dataFileString = input[iter];
			labelFileString = input[iter + 1];
			trainingSize = stoi(input[iter + 2]);
			learningRate = stod(input[iter + 3]);
			break;
		}
	}

	if (input.size() < 5)
	{
		cout << "Expected at least 4 parameters, use help for more information." << endl;
		return;
	}

	if (theNeuralNet.OutputSize == 0)
	{
		cout << "The neural network has not been set up." << endl;
	}

	ifstream dataFile;
	dataFile.open(dataFileString, ios::out | ios::binary);
	ifstream labelFile;
	labelFile.open(labelFileString, ios::out | ios::binary);

	uint32_t dataMagicNum = 2051;
	uint32_t labelMagicNum = 2049;

	int numOfImages;
	int numOfLabels;
	int rows;
	int cols;

	uint32_t a;
	while (dataFile.read(reinterpret_cast<char *>(&a), sizeof(a)))
	{
		a = _byteswap_ulong(a);
		if (a == dataMagicNum)
		{
			dataFile.read(reinterpret_cast<char *>(&a), sizeof(a));
			numOfImages = _byteswap_ulong(a);

			dataFile.read(reinterpret_cast<char *>(&a), sizeof(a));
			rows = _byteswap_ulong(a);

			dataFile.read(reinterpret_cast<char *>(&a), sizeof(a));
			cols = _byteswap_ulong(a);
			break;
		}
	}

	while (labelFile.read(reinterpret_cast<char *>(&a), sizeof(a)))
	{
		a = _byteswap_ulong(a);
		if (a == labelMagicNum)
		{
			labelFile.read(reinterpret_cast<char *>(&a), sizeof(a));
			numOfLabels = _byteswap_ulong(a);
			break;
		}
	}

	if (numOfImages != numOfLabels)
	{
		cout << "The training data files do not align." << endl;
		return;
	}
	else
		cout << "Reading in " << numOfImages << " total images to train with." << endl;

	vector<vector<double>> images;
	vector<vector<double>> labels;

	uchar b;

	//cv::Mat matimage = cv::Mat::zeros(cv::Size(cols, rows), CV_8U);

	for (int i = 0; i < numOfImages; i++)
	{
		vector<double> image;
		vector<double> label;
		for (int x = 0; x < cols; x++)
		{
			for (int y = 0; y < rows; y++)
			{
				dataFile.read(reinterpret_cast<char *>(&b), sizeof(b));
				image.push_back(double(b)/255);

				//matimage.at<uchar>(x, y) = b;
			}
		}
		//cv::imshow("test", matimage);
		//cvWaitKey(0);
		labelFile.read(reinterpret_cast<char *>(&b), sizeof(b));
		for (int j = 0; j < int(b); j++)
			label.push_back(0);
		label.push_back(1);
		for (size_t j = label.size(); j < 10; j++)
			label.push_back(0);

		images.push_back(image);
		labels.push_back(label);
	}

	cout << "Starting to train, press 'q' to stop." << endl;//TODO: add the q part
	int iteration = 0;
	for (int i = startAt; i < numOfImages; i += stepSize)
	{
		vector<vector<double>> inputImages;
		vector<vector<double>> inputLabels;
		for (int j = 0; j < trainingSize; j++)
		{
			int index = j + i;
			if (index > numOfImages)
				index -= numOfImages;
			inputImages.push_back(images[index]);
			inputLabels.push_back(labels[index]);
		}
		double cost = theNeuralNet.BackPropagate(inputImages, inputLabels, learningRate);
		if(printModular != 0 && iteration % printModular == 0)
			cout << "Cost:\t" << cost << endl;
		iteration++;
	}

}

uint32_t readuint32FromFile(ifstream stream)
{
	uint32_t a;
	stream.read(reinterpret_cast<char *>(&a), sizeof(a));
	return(_byteswap_ulong(a));
}

vector<double> LoadImage(string filePath)
{
	cv::Mat image = cv::imread(filePath, cv::IMREAD_GRAYSCALE);
	vector<double> imageVector;

	for (int x = 0; x < image.cols; x++)
		for (int y = 0; y < image.rows; y++)
			imageVector.push_back(double(image.at<uchar>(x, y))/255);
	/*
	cv::Mat matimage = cv::Mat::zeros(cv::Size(28, 28), CV_8U);
	int i = 0;
	for (int x = 0; x < 28; x++)
	{
		for (int y = 0; y < 28; y++)
		{
			uchar b = imageVector[i] * 255;
			i++;
			matimage.at<uchar>(x, y) = b;
		}
	}
	cv::imshow("test", matimage);
	cvWaitKey(0);
	//*/
	return imageVector;
}
