#include "AiModel.hpp"

namespace nn {
AiModel::AiModel(const std::string &config_file, const bool use_visual)
    : config(config_file) {
	_model = std::make_unique<model>(config, use_visual);
}

void AiModel::run_model(const std::vector<Global::ValueType> &input) {
	_model->run_model(input);
}

prediction AiModel::getPrediction() {
	size_t max = 0;

	for (size_t i = 1; i < _model->getOutputSize(); i++) {
		if (_model->getOutput()[i] > _model->getOutput()[max])
			max = i;
	}

	return {max, _model->getOutput()[max]};
}
} // namespace nn
