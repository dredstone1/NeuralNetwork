#include "model.hpp"
#include "FNNetwork.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

namespace nn::model {
Model::Model(const std::string &config_filepath) {
	loadConfig(config_filepath);
	visual.start();
}

void Model::runModel(const global::ParamMetrix &input) {
	network[0]->forward(input);

	for (size_t i = 1; i < network.size(); i++) {
		network[i]->forward(network[i - 1]->getOutput());
	}
}

void Model::loadConfig(const std::string &config_filepath) {
	std::ifstream ifs(config_filepath);
	if (!ifs.is_open()) {
		std::cerr << "Error: Could not open config file: " << config_filepath << std::endl;
		throw std::runtime_error("Failed to open config file: " + config_filepath);
	}

	nlohmann::json j;
	try {
		ifs >> j;
		const auto &networkConifg = j.at("network");

		for (auto &subNetworkConfig : networkConifg) {
			std::string type = subNetworkConfig.at("type");
            if (type == "dense"){
                network.push_back(std::make_unique<DenseNetwork>(subNetworkConfig));
            }
		}
	} catch (const nlohmann::json::parse_error &e) {
		std::cerr << "JSON parse error in file '" << config_filepath << "':\n"
		          << e.what() << "\n"
		          << "at byte " << e.byte << std::endl;
		throw;
	} catch (const nlohmann::json::exception &e) {
		std::cerr << "JSON processing error in file '" << config_filepath << "':\n"
		          << e.what() << std::endl;
		throw;
	}
}

void Model::resetNetworkGradient() {
	for (auto &subNet : network) {
		subNet->resetGradient();
	}
}

void Model::update_weights(const int batch_size) {
	const global::ValueType CURRENT_LEARNING_RATE = -learningRate.currentLearningRate / batch_size;
	for (auto &subNet : network) {
		subNet->updateWeights(CURRENT_LEARNING_RATE);
	}
}

void Model::Backward(const global::ParamMetrix &output) {
	global::ParamMetrix deltas = output;
	global::ParamMetrix temp = output;

	network[network.size() - 1]->backword(deltas, temp);
	deltas = temp;

	for (int i = network.size() - 2; i >= 0; i--) {
		network[i]->backword(deltas, temp);
		deltas = temp;
	}
}

global::ValueType Model::run_back_propagation(const Batch &batch) {
	global::ValueType error = 0.0;

	if (batch.size() == 0) {
		return error;
	}

	resetNetworkGradient();
	for (size_t i = 0; i < batch.size(); i++) {
		const TrainSample *current_sample_ptr = batch.samples.at(i);

		visual.updatePrediction(current_sample_ptr->prediction.index);
		runModel(current_sample_ptr->input);
		global::ParamMetrix output(inputSize(), 0);
		output[current_sample_ptr->prediction.index] = 1;
		Backward(output);
		error += getLost(output);

		update_weights(batch.size());
	}
	return error / batch.size();
}

void Model::train() {
	std::cout << "Training AI" << std::endl;

	const auto start = std::chrono::high_resolution_clock::now();
	global::ValueType error = 0.0;

	visual.updateAlgoritemMode(visualizer::AlgorithmMode::Training);
	visual.updateLearningRate(learningRate.currentLearningRate);

	for (int loop_index = 0; loop_index < config.training_config.batch_count + 1; loop_index++) {
		visual.updateBatchCounter(loop_index);

		Batch &batch = dataBase.get_Batch();
		error = run_back_propagation(batch);

		visual.updateError(error, loop_index);

		// print_progress_bar(loop_index + 1, config.training_config.batch_count);

		visual.updateLearningRate(learningRate.currentLearningRate);
		if (visual.exit_training() == true)
			break;
	}

	const auto end = std::chrono::high_resolution_clock::now();
	const int time_taken = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
	const int minutes = time_taken / SECONDS_IN_MINUTE;
	const int seconds = time_taken % SECONDS_IN_MINUTE;
	const int time_taken_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << std::endl
	          << "Training Done!" << std::endl
	          << "Training time: "
	          << minutes << " minutes "
	          << seconds << " seconds" << " ("
	          << time_taken_milliseconds << " ms)" << std::endl
	          << "final_score: " << error << std::endl;

	visual.updateAlgoritemMode(visualizer::AlgorithmMode::Normal);
}

void Model::reset() {
	for (auto &subNetwork : network) {
		subNetwork.reset();
	}
}

const global::ParamMetrix &Model::getOutput() const {
	return network[network.size() - 1]->getOutput();
}

global::ValueType Model::getLost(const global::ParamMetrix &output) {
	return network[network.size() - 1]->getLost(output);
}

int Model::outputSize() {
	return network[network.size() - 1]->outputSize();
}

int Model::inputSize() {
	return network[0]->inputSize();
}

void Model::updateWeights(const global::ValueType learningRate) {
	visual.setNewPhaseMode(visualizer::NnMode::Backward);

	for (int i = network.size() - 1; i >= 0; i--) {
		network[i]->updateWeights(learningRate);
	}

	visual.setNewPhaseMode(visualizer::NnMode::Forword);
}
} // namespace nn::model
