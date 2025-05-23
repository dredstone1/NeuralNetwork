#include "../include/AiPlayer/AiModel.hpp"
#include "AiPlayer/AiPlayer.hpp"
#include "Layers/layer.hpp"
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <string>
#include <vector>

AiModel::AiModel() {
	_model = new model(9, 9, 5, 2);
}

AiModel::AiModel(string _file_name) { load(_file_name); }

int AiModel::load(const string file_name) {
	ifstream file(file_name + ".model");

	if (!file.is_open()) {
		cout << "File not found" << endl;
		return 1;
	}

	string line;
	getline(file, line);
	file.close();
	char *line_ = &line[0];

	// TODO: add error handling.
	int input_size = atoi(strtok(line_, " "));
	int output_size = atoi(strtok(NULL, " "));
	int hidden_layers_size = atoi(strtok(NULL, " "));
	int hidden_layers_count = atoi(strtok(NULL, " "));

	delete _model;
	_model = new model(input_size, output_size, hidden_layers_size, hidden_layers_count);

	for (int i = 0; i < _model->getLayerCount(); i++) {
		Layer &layer = _model->getLayer(i);
		for (int j = 0; j < layer.getSize(); j++) {
			layer.setBias(j, atof(strtok(NULL, " ")));

			for (int k = 0; k < layer.getPrevSize(); k++) {
				layer.setWeight(j, k, atof(strtok(NULL, " ")));
			}
		}
	}

	return 0;
}

void AiModel::save(const string file_name) {
	ostringstream oss;
	oss << fixed << setprecision(10)
	    << _model->getInputSize() << " "
	    << _model->getOutputSize() << " "
	    << _model->getHiddenLayerSize() << " "
	    << _model->getHiddenLayerCount() << " ";

	for (int i = 0; i < _model->getLayerCount(); i++) {
		Layer &layer = _model->getLayer(i);
		for (int j = 0; j < layer.getSize(); j++) {
			oss << layer.getBias(j) << " ";

			for (int k = 0; k < layer.getPrevSize(); k++) {
				oss << layer.getWeight(j, k) << " ";
			}
		}
	}

	ofstream file(file_name + ".model");
	file << oss.str();
	file.close();
}

void AiModel::run_model(const vector<double> &input) {
	_model->run_model(input);
}

int AiModel::getPrediction(const vector<double> &input) {
	int max1 = 0, max = 0;
	for (int i = 0; i < 9; i++) {
		if (_model->getOutput()[max1] < _model->getOutput()[i]) {
			max1 = i;
		}
	}

	cout << "max1: " << max1 << endl;
	while (input[max1] != CELL_VALUE_EMPTY) {
		if (max1 == 8 || max == 1) {
			max1--;
			max = 1;
			continue;
		}

		max1++;
	}

	// print the output of the model.
	for (int i = 0; i < 9; i++) {
		cout << "Net: " << _model->getLayer(_model->getLayerCount() - 1).getNet()[i] << " ";
	}
	cout << endl;

	for (int i = 0; i < 9; i++) {
		cout << "Out: " << _model->getLayer(_model->getLayerCount() - 1).getOut()[i] << " ";
	}
	cout << endl;

	return max1;
}

AiModel::~AiModel() { 
    delete _model; 
}
