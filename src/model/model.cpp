#include "model.hpp"
#include "FNNetwork.hpp"
#include "dataBase.hpp"
#include <fstream>
#include <iostream>

namespace nn::model {
Model::Model(const std::string &config_filepath)
    : config(config_filepath),
      visual(config),
      learningRate(config.trainingConfig.lr_init_value) {
	initModel();
	initVisual();
}

void Model::initVisual() {
	visual.start();

	for (size_t i = 0; i < config.networkConfig.SubNetworksConfig.size(); ++i) {
		auto _config = config.networkConfig.SubNetworksConfig[i];

		if (_config->NNLable() == "FNN") {
			visual.addVisualSubNetwork(network[i]->getVisual());
		}
	}
}

void Model::initModel() {
	const std::uint32_t width = visualizer::SUB_NETWORKS_WIDTH / config.networkConfig.SubNetworksConfig.size();

	for (size_t i = 0; i < config.networkConfig.SubNetworksConfig.size(); ++i) {
		auto _config = config.networkConfig.SubNetworksConfig[i];

		if (_config->NNLable() == "FNN") {
			FNNConfig &sub_ = *dynamic_cast<FNNConfig *>(_config.get());
			std::shared_ptr<visualizer::FnnVisualier> visual_ = std::make_shared<visualizer::FnnVisualier>(visual.Vstate, width, sub_);

			network.push_back(std::make_unique<FNNetwork>(sub_, true, visual_));
		}
	}
}

void Model::runModel(const global::ParamMetrix &input) {
	visual.updateInput(input);
	network[0]->forward(input);

	for (size_t i = 1; i < network.size(); ++i) {
		network[i]->forward(network[i - 1]->getOutput());
	}
}

void Model::resetNetworkGradient() {
	for (auto &subNet : network) {
		subNet->resetGradient();
	}
}

void Model::update_weights(const int batch_size) {
	const global::ValueType CURRENT_LEARNING_RATE = learningRate / batch_size;

	for (auto &subNet : network) {
		subNet->updateWeights(CURRENT_LEARNING_RATE);
	}
}

void Model::Backward(const global::ParamMetrix &output) {
	global::ParamMetrix deltas = output;

	for (int i = static_cast<int>(network.size()) - 1; i >= 0; --i) {
		network[i]->backward(deltas);
		deltas = network[i]->getInput();
	}
}

global::ValueType Model::run_back_propagation(const Batch &batch, const bool doBackward) {
	global::ValueType error = 0.0;

	if (batch.size() == 0) {
		return error;
	}

	resetNetworkGradient();
	for (size_t i = 0; i < batch.size(); ++i) {
		auto current_sample_ptr = batch.samples.at(i);
		visual.updatePrediction(current_sample_ptr->pre);
		runModel(current_sample_ptr->input);
		global::ParamMetrix output(outputSize());
		output[current_sample_ptr->pre.index] = 1;

		if (doBackward) {
			Backward(output);
			update_weights(batch.size());
		}

		error += getLoss(current_sample_ptr->pre);
	}

	return error / batch.size();
}

void Model::print_progress_bar(const int current, const int total) {
	float progress = (float)current / total;
	int progress_percentage = int(progress * BAR_WIDTH);

	if (progress_percentage != lastProgress) {
		int pos = BAR_WIDTH * progress;
		lastProgress = progress_percentage;

		std::ostringstream oss;
		oss << "[";
		for (int i = 0; i < BAR_WIDTH; ++i) {
			if (i < pos)
				oss << "=";
			else if (i == pos)
				oss << ">";
			else
				oss << " ";
		}
		oss << "] " << progress_percentage << " %\r";

		std::cout << oss.str();
		std::cout.flush();
	}
}

void Model::printTrainingResult(const std::chrono::high_resolution_clock::time_point &start, double error) {
	const auto end = std::chrono::high_resolution_clock::now();
	const int time_taken = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
	const int minutes = time_taken / SECONDS_IN_MINUTE;
	const int seconds = time_taken % SECONDS_IN_MINUTE;
	const int time_taken_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	std::cout << "\n"
	          << "Training Done!" << "\n"
	          << "Training time: "
	          << minutes << " minutes "
	          << seconds << " seconds" << " ("
	          << time_taken_milliseconds << " ms)" << "\n"
	          << "final score: " << error << "\n";
}

void Model::train(const std::string &db_filename) {
	DataBase dataBase(config.trainingConfig);
	const auto start = std::chrono::high_resolution_clock::now();
	global::ValueType error = 0.0;

	std::cout << "Training AI" << std::endl;

	dataBase.load(db_filename);

	visual.updateAlgorithmMode(visualizer::AlgorithmMode::Training);
	visual.updateLearningRate(learningRate);

	for (int i = 0; i < config.trainingConfig.batch_count + 1; ++i) {
		visual.updateBatchCounter(i);

		Batch &batch = dataBase.getBatch();
		error = run_back_propagation(batch, true);

		if (config.trainingConfig.save_every > 0 && i % config.trainingConfig.save_every == 0) {
			save(config.trainingConfig.data_filename_autoSave);
		}

		visual.updateError(error, i);

		print_progress_bar(i + 1, config.trainingConfig.batch_count);

		visual.updateLearningRate(learningRate);
		if (visual.exitTraining() == true) {
			break;
		}
	}

	printTrainingResult(start, error);

	visual.updateAlgorithmMode(visualizer::AlgorithmMode::Normal);
}

modelResult Model::evaluateModel(const std::string &db_filename) {
	modelResult result;
	DataBase dataBase(config.trainingConfig);

	std::cout << "Evaluating AI" << std::endl;

	dataBase.load(db_filename);
	result.dbSize = dataBase.DataBaseLength();

	for (int i = 0; i < result.dbSize; ++i) {
		TrainSample &sample = dataBase.getSample(i);
		runModel(sample.input);

		size_t predicted_index = std::distance(
		    getOutput().begin(),
		    std::max_element(getOutput().begin(), getOutput().end()));

		if (predicted_index == sample.pre.index) {
			result.currectPreSize++;
		}
	}

	result.percentage = 100.f * result.currectPreSize / result.dbSize;

	return result;
}

const global::ParamMetrix &Model::getOutput() const {
	return network[network.size() - 1]->getOutput();
}

global::ValueType Model::getLoss(const global::Prediction &pre) {
	return network[network.size() - 1]->getLoss(pre);
}

int Model::outputSize() {
	return network[network.size() - 1]->outputSize();
}

int Model::inputSize() {
	return network[0]->inputSize();
}

void Model::updateWeights(const global::ValueType learningRate) {
	visual.setNewPhaseMode(visualizer::NnMode::Backward);

	for (int i = network.size() - 1; i >= 0; --i) {
		network[i]->updateWeights(learningRate);
	}

	visual.setNewPhaseMode(visualizer::NnMode::Forword);
}

void Model::save(const std::string &file) {
	std::ofstream outFile(file);

	for (size_t i = 0; i < network.size(); ++i) {
		global::ParamMetrix params = network[i]->getParams();

		for (size_t j = 0; j < params.size(); ++j) {
			outFile << params[j] << " ";
		}

		outFile << std::endl;
	}

	outFile.close();
}

void Model::load(const std::string &file) {
	std::ifstream inFile(file);

	std::string line;
	int networkI = 0;
	while (std::getline(inFile, line)) {
		std::istringstream iss(line);
		global::ParamMetrix numbers;
		float num;

		while (iss >> num) {
			numbers.push_back(num);
		}

		network[networkI]->setParams(numbers);

		networkI++;
	}

	inFile.close();
}
} // namespace nn::model
