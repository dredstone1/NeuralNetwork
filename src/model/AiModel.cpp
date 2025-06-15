#include "AiModel.hpp"

namespace nn {
AiModel::AiModel(const std::string &config_file)
    : config(config_file) {
	model = std::make_unique<model::Model>(config);
}

void AiModel::runModel(const global::ParamMetrix &input) {
	model->runModel(input);
}

Prediction AiModel::getPrediction() {
	size_t max = 0;

	for (size_t i = 1; i < model->getOutputSize(); i++) {
		if (model->getOutput()[i] > model->getOutput()[max]) {
			max = i;
		}
	}

	return {max, model->getOutput()[max]};
}
} // namespace nn
