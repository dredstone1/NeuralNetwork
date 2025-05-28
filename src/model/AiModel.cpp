#include "../include/AiModel.hpp"
#include "model/config.hpp"
#include "model/model.hpp"
#include <cstring>
#include <string>
#include <vector>

AiModel::AiModel(std::string config_file, const bool use_visual) : config(config_file){
	_model = new model(config, use_visual);
}

void AiModel::run_model(const std::vector<double> &input) {
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
