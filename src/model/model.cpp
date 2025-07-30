#include "../networks/fnn/FNNetwork.hpp"
#include "dataBase.hpp"
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <model.hpp>

namespace nn::visualizer {
void ProgressBar::printBar() {
	if (total == 0)
		return;

	int percentage = 100.0 * current / total;

	if (percentage == last_percentage)
		return;

	last_percentage = percentage;

	const int pos = BAR_WIDTH * current / total;

	char bar[BAR_WIDTH + 64];
	int index = 0;

	bar[index++] = '[';
	for (int i = 0; i < BAR_WIDTH; ++i) {
		if (i < pos)
			bar[index++] = '=';
		else if (i == pos)
			bar[index++] = '>';
		else
			bar[index++] = ' ';
	}
	bar[index++] = ']';
	bar[index++] = ' ';

	int written = std::snprintf(bar + index, sizeof(bar) - index, "%3d %%", percentage);
	if (written > 0)
		index += written;

	bar[index++] = (percentage == 100) ? '\n' : '\r';
	bar[index] = '\0';

	std::cout << header << bar << std::flush;
}

ProgressBar ProgressBar::operator++(int) {
	ProgressBar temp = *this;
	++current;
	if (current > total)
		current = total;
	return temp;
}
}; // namespace nn::visualizer

namespace nn::model {
Model::Model(const std::string &config_filepath)
    : config(config_filepath),
      visual(config),
      learningRate(config.trainingConfig.getLearningRate()) {
	initOptimizer();
	initModel();
	if (config.visualConfig.enableVisuals) {
		initVisual();
	}
}

void Model::initOptimizer() {
	const std::string &type = config.trainingConfig.getOptimizerType();

	if (type == "const") {
		auto *optConfig = dynamic_cast<ConstantOptimizerConfig *>(config.trainingConfig.getOptimizer().get());
		optimizer = std::make_shared<ConstantOptimizer>(*optConfig);
	}
}

void Model::initVisual() {
	visual.start();

	for (size_t i = 0; i < config.networkConfig.SubNetworksConfig.size(); ++i) {
		auto _config = config.networkConfig.SubNetworksConfig[i];

		if (_config->NNLable() == "FNN") {
			visual.addVisualSubNetwork(network[i]->getVisual());
			network[i]->getVisual()->setVstate(visual.Vstate);
		}
	}
}

void Model::initModel() {
	const std::uint32_t width = visualizer::SUB_NETWORKS_WIDTH / config.networkConfig.SubNetworksConfig.size();

	for (size_t i = 0; i < config.networkConfig.SubNetworksConfig.size(); ++i) {
		auto _config = config.networkConfig.SubNetworksConfig[i];

		if (_config->NNLable() == "FNN") {
			FNNConfig &sub_ = *dynamic_cast<FNNConfig *>(_config.get());

			if (config.visualConfig.enableVisuals) {
				std::shared_ptr<visualizer::fnn::FnnVisualier> visual_ =
				    std::make_shared<visualizer::fnn::FnnVisualier>(
				        visual.Vstate,
				        width,
				        sub_);

				network.push_back(std::make_unique<fnn::FNNetwork>(sub_, true, visual_));
			} else {
				network.push_back(std::make_unique<fnn::FNNetwork>(sub_, true));
			}
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

void Model::updateWeights(const int batchSize) {
	optimizer->setOfset(batchSize);

	for (auto &subNet : network) {
		subNet->updateWeights(optimizer);
	}
}

void Model::Backward(const global::ParamMetrix &output) {
	global::ParamMetrix deltas = output;

	for (int i = static_cast<int>(network.size()) - 1; i >= 0; --i) {
		network[i]->backward(deltas);
		deltas = network[i]->getInput();
	}
}

global::ValueType Model::runBackPropagation(
    const Batch &batch,
    const bool doBackward,
    global::Transformation transformation) {
	global::ValueType error = 0.0;

	if (batch.size() == 0) {
		return error;
	}

	resetNetworkGradient();
	for (size_t i = 0; i < batch.size(); ++i) {
		auto current_sample_ptr = batch.samples.at(i);
		visual.updatePrediction(current_sample_ptr->pre);
		if (transformation == nullptr) {
			runModel(current_sample_ptr->input);
		} else {
			global::ParamMetrix newSample = transformation(current_sample_ptr->input);
			runModel(newSample);
		}

		global::ParamMetrix output(outputSize());
		output[current_sample_ptr->pre.index] = 1;

		if (doBackward) {
			Backward(output);
			updateWeights(batch.size());
		}

		error += getLoss(current_sample_ptr->pre);
	}

	return error / batch.size();
}

void Model::printTrainingResult(
    const std::chrono::high_resolution_clock::time_point &start,
    const double error) {
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

void Model::train(
    const std::string &db_filename,
    global::Transformation transformation) {
	DataBase trainedDataBase(config.trainingConfig);
	DataBase evaluateDataBase(config.trainingConfig);

	if (config.trainingConfig.isAutoEvaluating()) {
		evaluateDataBase.load(config.trainingConfig.getAutoEvaluating().dataBaseFilename);
	}

	std::cout << "Training AI" << std::endl;

	trainedDataBase.load(db_filename);

	trainModel(trainedDataBase, evaluateDataBase, transformation);
}

void Model::train(
    const std::vector<std::string> &db_filename,
    global::Transformation transformation) {
	DataBase trainedDataBase(config.trainingConfig);
	DataBase evaluateDataBase(config.trainingConfig);

	if (config.trainingConfig.isAutoEvaluating()) {
		evaluateDataBase.load(config.trainingConfig.getAutoEvaluating().dataBaseFilename);
	}

	std::cout << "Training AI" << std::endl;

	trainedDataBase.load(db_filename);

	trainModel(trainedDataBase, evaluateDataBase, transformation);
}

void Model::trainModel(
    DataBase &trainedDataBase,
    DataBase &evaluateDataBase,
    global::Transformation transformation) {
	visualizer::ProgressBar bar(config.trainingConfig.getBatchCount(), TRAINING_HEADER);

	const auto start = std::chrono::high_resolution_clock::now();
	global::ValueType error = 0.0;

	visual.updateLearningRate(learningRate);

	setTraining();
	for (int i = 0; i < config.trainingConfig.getBatchCount() + 1; ++i) {
		visual.updateBatchCounter(i);

		Batch &batch = trainedDataBase.getBatch();
		error = runBackPropagation(batch, true, transformation);
		visual.updateLost(error, i);

		if (config.trainingConfig.isAutoSave() && i % config.trainingConfig.getAutoSave().saveEvery == 0) {
			save(config.trainingConfig.getAutoSave().dataFilenameAutoSave);
		}

		setEvaluating();
		if (i >= config.trainingConfig.getAutoEvaluating().evaluateEvery &&
		    config.trainingConfig.isAutoEvaluating() &&
		    i % config.trainingConfig.getAutoEvaluating().evaluateEvery == 0) {

			modelResult result = evaluateModel(evaluateDataBase, false, false, transformation);
			visual.updateEvaluate(result.percentage, i);

			if (result.percentage == 100) {
				break;
			}
		}

		if (visual.exitTraining()) {
			break;
		}

		setTraining();

		bar++;
		bar.printBar();

		visual.updateLearningRate(learningRate);
	}
	setNormal();

	printTrainingResult(start, error);
}

float Model::calculatePercentage(size_t currentSize, size_t totalSize) {
	if (totalSize == 0)
		return 0.0f;
	return 100.0f * static_cast<float>(currentSize) / static_cast<float>(totalSize);
}

modelResult Model::evaluateModel(
    DataBase &dataBase,
    const bool cancleOnError,
    const bool showProgressbar,
    global::Transformation transformation) {
	modelResult result{0, 0, 0};

	if (showProgressbar) {
		std::cout << "Evaluating AI" << std::endl;
	}

	result.dbSize = dataBase.DataBaseLength();
	visualizer::ProgressBar bar(result.dbSize, EVALUATING_HEADER);

	setEvaluating();

	for (int i = 0; i < result.dbSize; ++i) {
		TrainSample &sample = dataBase.getSample(i);

		if (transformation == nullptr) {
			runModel(sample.input);
		} else {
			runModel(transformation(sample.input));
		}

		size_t predicted_index = std::distance(
		    getOutput().begin(),
		    std::max_element(getOutput().begin(), getOutput().end()));

		if (showProgressbar) {
			bar++;
			bar.printBar();
		}

		if (predicted_index == sample.pre.index) {
			result.currectPreSize++;
		} else if (cancleOnError) {
			break;
		}
	}

	result.percentage = calculatePercentage(result.currectPreSize, result.dbSize);

	setNormal();
	return result;
}

modelResult Model::evaluateModel(
    const std::string &db_filename,
    const bool cancleOnError,
    global::Transformation transformation) {
	DataBase dataBase(config.trainingConfig);
	dataBase.load(db_filename);
	return evaluateModel(dataBase, cancleOnError, transformation);
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

global::Prediction Model::getPrediction() {
	int max = 0;

	for (int i = 1; i < (int)outputSize(); ++i) {
		if (getOutput()[i] > getOutput()[max]) {
			max = i;
		}
	}

	return {max, getOutput()[max]};
}

void Model::setTraining() {
	visual.updateAlgorithmMode(visualizer::AlgorithmMode::Training);

	for (auto &sub : network) {
		sub->setTraining(true);
	}
}

void Model::setNormal() {
	visual.updateAlgorithmMode(visualizer::AlgorithmMode::Normal);

	for (auto &sub : network) {
		sub->setTraining(false);
	}
}
void Model::setEvaluating() {
	visual.updateAlgorithmMode(visualizer::AlgorithmMode::Evaluating);

	for (auto &sub : network) {
		sub->setTraining(false);
	}
}
} // namespace nn::model
