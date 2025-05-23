#include "../include/AiModel.hpp"
#include "Layers/layer.hpp"
#include "model/model.hpp"
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <string>
#include <vector>

AiModel::AiModel(const bool use_visual) {
	_model = new model(9, 9, 5, 2, use_visual);
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

	bool useVisual = _model->useVisual;
	delete _model;
	_model = new model(input_size, output_size, hidden_layers_size, hidden_layers_count, useVisual);

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

prediction AiModel::getPrediction() {
	int max = 0;

	for (int i = 1; i < _model->getOutputSize(); i++) {
		if (_model->getOutput()[i] > _model->getOutput()[max])
			max = i;
	}

	return {max, _model->getOutput()[max]};
}

AiModel::~AiModel() {
	delete _model;
}
