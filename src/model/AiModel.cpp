#include "AiModel.hpp"
#include "model.hpp"

namespace nn {
AiModel::AiModel(const std::string &config_file) {
	model = std::make_unique<model::Model>(config_file);
}

void AiModel::runModel(const global::ParamMetrix &input) {
	model->runModel(input);
}

void AiModel::train(const std::string &db_filename) {
	model->train(db_filename);
}

global::FinalPrediction AiModel::getPrediction() {
	int max = 0;

	for (int i = 1; i < (int)model->outputSize(); ++i) {
		if (model->getOutput()[i] > model->getOutput()[max]) {
			max = i;
		}
	}

	return {max, model->getOutput()[max]};
}

void AiModel::save(const std::string &file) {
	model->save(file);
}

model::modelResult AiModel::evaluateModel(const std::string &db_filename) {
	return model->evaluateModel(db_filename);
}

void AiModel::load(const std::string &file) {
	model->load(file);
}
} // namespace nn
