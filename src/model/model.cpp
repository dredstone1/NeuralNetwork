#include "model.hpp"
#include "FNNetwork.hpp"
#include "dataBase.hpp"
#include <fstream>
#include <iostream>
#include <iterator>

namespace nn::model {
Model::Model(const std::string &config_filepath)
    : config(config_filepath),
      visual(config),
      learningRate(config.trainingConfig.getLearningRate()) {
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

global::ValueType Model::runBackPropagation(const Batch &batch, const bool doBackward) {
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

void ProgressBar::printBar() {
	if (total == 0)
		return;

	const int percentage = 100 * current / total;
	if (percentage == last_percentage)
		return;

	last_percentage = percentage;

	const int pos = BAR_WIDTH * current / total;

	char bar[BAR_WIDTH + 64];
	int index = 0;

	bar[index++] = '[';
	for (int i = 0; i < BAR_WIDTH; ++i)
		bar[index++] = (i < pos) ? '=' : (i == pos) ? '>'
		                                            : ' ';
	bar[index++] = ']';
	bar[index++] = ' ';

	if (percentage < 10) {
		bar[index++] = ' ';
		bar[index++] = ' ';
		bar[index++] = percentage + '0';
	} else if (percentage < 100) {
		bar[index++] = ' ';
		bar[index++] = (percentage / 10) + '0';
		bar[index++] = (percentage % 10) + '0';
	} else {
		bar[index++] = '1';
		bar[index++] = '0';
		bar[index++] = '0';
	}

	bar[index++] = ' ';
	bar[index++] = '%';

	bar[index++] = (percentage == 100) ? '\n' : '\r';
	bar[index] = '\0';

	std::cout << header << bar << std::flush;
}

void Model::printTrainingResult(const std::chrono::high_resolution_clock::time_point &start, double error) {
	const auto end = std::chrono::high_resolution_clock::now();
	const int time_taken = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
	const int minutes = time_taken / SECONDS_IN_MINUTE;
	const int seconds = time_taken % SECONDS_IN_MINUTE;
	const int time_taken_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	std::cout << "Training Done!" << "\n"
	          << "Training time: "
	          << minutes << " minutes "
	          << seconds << " seconds" << " ("
	          << time_taken_milliseconds << " ms)" << "\n"
	          << "final score: " << error << "\n";
}

void Model::train(const std::string &db_filename) {
	DataBase trainedDataBase(config.trainingConfig);
	DataBase evaluateDataBase(config.trainingConfig);

	const auto start = std::chrono::high_resolution_clock::now();
	global::ValueType error = 0.0;
	ProgressBar bar(config.trainingConfig.getBatchCount(), TRAINING_HEADER);

	if (config.trainingConfig.isAutoEvaluating()) {
		evaluateDataBase.load(config.trainingConfig.getAutoEvaluating().dataBaseFilename);
	}

	std::cout << "Training AI" << std::endl;

	trainedDataBase.load(db_filename);

	visual.updateAlgorithmMode(visualizer::AlgorithmMode::Training);
	visual.updateLearningRate(learningRate);

	for (int i = 0; i < config.trainingConfig.getBatchCount() + 1; ++i) {
		visual.updateBatchCounter(i);

		Batch &batch = trainedDataBase.getBatch();
		error = runBackPropagation(batch, true);

		if (config.trainingConfig.isAutoSave() && i % config.trainingConfig.getAutoSave().saveEvery == 0) {
			save(config.trainingConfig.getAutoSave().dataFilenameAutoSave);
		}

		if (config.trainingConfig.isAutoEvaluating() && i % config.trainingConfig.getAutoEvaluating().evaluateEvery == 0) {
			modelResult result = evaluateModel(evaluateDataBase, true);

			if (result.percentage == 100) {
				break;
			}
		}

		visual.updateError(error, i);

		bar++;
		bar.printBar();

		visual.updateLearningRate(learningRate);
		if (visual.exitTraining() == true) {
			break;
		}
	}

	printTrainingResult(start, error);

	visual.updateAlgorithmMode(visualizer::AlgorithmMode::Normal);
}

float Model::calculatePercentage(size_t currentSize, size_t totalSize) {
	if (totalSize == 0)
		return 0.0f;
	return 100.0f * static_cast<float>(currentSize) / static_cast<float>(totalSize);
}

modelResult Model::evaluateModel(DataBase &dataBase, const bool cancleOnError) {
	modelResult result;

	std::cout << "Evaluating AI" << std::endl;

	result.dbSize = dataBase.DataBaseLength();
	ProgressBar bar(result.dbSize, EVALUATING_HEADER);

	for (int i = 0; i < result.dbSize; ++i) {
		TrainSample &sample = dataBase.getSample(i);
		runModel(sample.input);

		size_t predicted_index = std::distance(
		    getOutput().begin(),
		    std::max_element(getOutput().begin(), getOutput().end()));

		bar++;
		bar.printBar();

		if (predicted_index == sample.pre.index) {
			result.currectPreSize++;
		} else if (cancleOnError) {
			break;
		}
	}

	result.percentage = calculatePercentage(result.currectPreSize, result.dbSize);
	return result;
}

modelResult Model::evaluateModel(const std::string &db_filename, const bool cancleOnError) {
	DataBase dataBase(config.trainingConfig);
	dataBase.load(db_filename);
	return evaluateModel(dataBase, cancleOnError);
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
