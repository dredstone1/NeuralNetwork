#include "../include/AiModel.hpp"
#include "model/config.hpp"
#include "model/model.hpp"
#include <cstring>
#include <string>
#include <vector>

AiModel::AiModel(string config_file, const bool use_visual) {
	config = new Config(config_file);
	_model = new model(*config, use_visual);
}

int AiModel::load(const string file_name, bool use_visual) {
	return 0;
}

void AiModel::save(const string file_name) {
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
	delete config;
}
