#include "../networks/cnn/CNNetwork.hpp"
#include "../networks/fnn/FNNetwork.hpp"
#include "ProgressBar.hpp"
#include <fstream>
#include <iostream>
#include <model.hpp>

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
		optimizer = std::make_unique<ConstantOptimizer>(*optConfig);
	}
}

void Model::initVisual() {
	visual.start();

	if (!config.visualConfig.enableNetwrokVisual) {
		return;
	}

	for (size_t i = 0; i < config.networkConfig.SubNetworksConfig.size(); ++i) {
		visual.addVisualSubNetwork(network[i]->getVisual());
		network[i]->getVisual()->setVstate(visual.Vstate);
	}
}

std::uint32_t Model::calculateSubNetWidth() const {
	const auto count = config.networkConfig.SubNetworksConfig.size();
	if (count == 0) {
		return 0;
	}
	return visualizer::SUB_NETWORKS_WIDTH / static_cast<std::uint32_t>(count);
}

void Model::initModel() {
	const std::uint32_t WIDTH = calculateSubNetWidth();
	size_t param_amount = 0;

	for (size_t i = 0; i < config.networkConfig.SubNetworksConfig.size(); ++i) {
		ISubNetworkConfig &_config = *config.networkConfig.SubNetworksConfig[i];

		if (_config.NNLable() == fnn::FNN_LABLE) {
			addFNN(WIDTH, _config);
		} else if (_config.NNLable() == cnn::CNN_LABLE) {
			addCNN(WIDTH, _config);
		}

		param_amount += network[i]->getParamCount();
	}

	std::cout << "initialize model - "
	          << param_amount << " parameters, "
	          << config.networkConfig.SubNetworksConfig.size() << " sub networks"
	          << std::endl;
}

void Model::addFNN(const std::uint32_t width, ISubNetworkConfig &_config) {
	fnn::FNNConfig &sub_ = (fnn::FNNConfig &)(_config);

	if (config.visualConfig.enableVisuals && config.visualConfig.enableNetwrokVisual) {
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

void Model::addCNN(const std::uint32_t width, ISubNetworkConfig &_config) {
	cnn::CNNConfig &sub_ = (cnn::CNNConfig &)(_config);

	if (config.visualConfig.enableVisuals && config.visualConfig.enableNetwrokVisual) {
		std::shared_ptr<visualizer::cnn::CnnVisualier> visual_ =
		    std::make_shared<visualizer::cnn::CnnVisualier>(
		        visual.Vstate,
		        width,
		        sub_);

		network.push_back(std::make_unique<cnn::CNNetwork>(sub_, true, visual_));
	} else {
		network.push_back(std::make_unique<cnn::CNNetwork>(sub_, true));
	}
}

void Model::runModel(const global::Tensor &input) {
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
		subNet->updateWeights(*optimizer);
	}
}

void Model::Backward(const global::Tensor &output) {
	global::Tensor deltas = output;
	global::Tensor *delta = &deltas;

	for (int i = static_cast<int>(network.size()) - 1; i >= 0; --i) {
		network[i]->backward(&delta);
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
	global::Tensor output({outputSize()});
	for (size_t i = 0; i < batch.size(); ++i) {
		TrainSample *current_sample_ptr = batch.samples.at(i);
		visual.updatePrediction(current_sample_ptr->pre);

		runModel(current_sample_ptr->input, transformation);

		if (doBackward) {
			output.zero();
			output.setValue(current_sample_ptr->pre.index, 1);
			Backward(output);
			updateWeights(batch.size());
		}

		error += getLoss(current_sample_ptr->pre);
	}

	return error / batch.size();
}

void Model::printTrainingResult(
    const std::chrono::high_resolution_clock::time_point &start, double error) {
	const auto end = std::chrono::high_resolution_clock::now();
	const auto diff = end - start;
	const auto duration_ms =
	    std::chrono::duration_cast<std::chrono::milliseconds>(diff);

	const auto total_seconds = duration_ms.count() / 1000;
	const auto minutes = total_seconds / 60;
	const auto seconds = total_seconds % 60;

	std::cout << "Training Done!\n";
	std::cout << "Training time: " << minutes << " minutes "
	          << seconds << " seconds ("
	          << duration_ms.count() << " ms)\n";
	std::cout << "Final score: " << error << "\n";
}

void Model::train(
    const std::string &db_filename, global::Transformation transformationB,
    global::Transformation transformationE) {
	DataBase trainedDataBase(config.trainingConfig);
	DataBase evaluateDataBase(config.trainingConfig);

	if (config.trainingConfig.isAutoEvaluating()) {
		evaluateDataBase.load(
		    config.trainingConfig.getAutoEvaluating().dataBaseFilename);
	}

	trainedDataBase.load(db_filename);

	trainModel(trainedDataBase, evaluateDataBase, transformationB,
	           transformationE);
}

void Model::train(
    const std::vector<std::string> &db_filename,
    global::Transformation transformationB,
    global::Transformation transformationE) {
	DataBase trainedDataBase(config.trainingConfig);
	DataBase evaluateDataBase(config.trainingConfig);

	if (config.trainingConfig.isAutoEvaluating()) {
		evaluateDataBase.load(
		    config.trainingConfig.getAutoEvaluating().dataBaseFilename);
	}

	trainedDataBase.load(db_filename);

	trainModel(trainedDataBase, evaluateDataBase, transformationB,
	           transformationE);
}

bool Model::autoEvaluating(
    const int i,
    DataBase &evaluateDataBase,
    global::Transformation transformationE) {
	setEvaluating();
	if (config.trainingConfig.isAutoEvaluating() &&
	    i % config.trainingConfig.getAutoEvaluating().evaluateEvery == 0) {
		modelResult result = evaluateModel(evaluateDataBase, false, false, transformationE);
		visual.updateEvaluate(result.percentage, i / config.trainingConfig.getAutoEvaluating().evaluateEvery);

		if (result.percentage == 100) {
			return true;
		}
	}

	return false;
}

void Model::autoSave(int i) {
	if (i <= 0) {
		return;
	}

	const auto &autoSaveCfg = config.trainingConfig.getAutoSave();
	if (!config.trainingConfig.isAutoSave() || i % autoSaveCfg.saveEvery != 0) {
		return;
	}

	save(autoSaveCfg.dataFilenameAutoSave, false);
}

void Model::trainModel(
    DataBase &trainedDataBase, DataBase &evaluateDataBase,
    global::Transformation transformationB,
    global::Transformation transformationE) {
	ProgressBar bar(config.trainingConfig.getBatchCount(), TRAINING_HEADER);

	const auto start = std::chrono::high_resolution_clock::now();
	global::ValueType error = 0.0;

	visual.updateLearningRate(learningRate);

	setTraining();
	for (size_t i = 0; i < config.trainingConfig.getBatchCount() + 1; ++i) {
		visual.updateBatchCounter(i);

		Batch &batch = trainedDataBase.getBatch();
		error = runBackPropagation(batch, true, transformationB);
		visual.updateLost(error, i);

		autoSave(i);

		if (visual.exitTraining() || autoEvaluating(i, evaluateDataBase, transformationE)) {
			break;
		}

		setTraining();

		bar++;
		bar.printBar();

		visual.updateLearningRate(learningRate);
	}
	setNormal();

	bar.endPrint();

	printTrainingResult(start, error);
}

float Model::calculatePercentage(size_t currentSize, size_t totalSize) {
	if (totalSize == 0) {
		return 0.0f;
	}

	return 100.0f * static_cast<float>(currentSize) / static_cast<float>(totalSize);
}

void Model::runModel(const global::Tensor &input,
                     global::Transformation transformation) {
	if (transformation) {
		runModel(transformation(input));
	} else {
		runModel(input);
	}
}

modelResult Model::evaluateModel(
    DataBase &dataBase,
    const bool cancleOnError,
    const bool showProgressbar,
    global::Transformation transformation) {
	modelResult result{0, 0, 0};

	result.dbSize = dataBase.DataBaseLength();
	ProgressBar bar(result.dbSize, EVALUATING_HEADER);

	setEvaluating();

	for (int i = 0; i < result.dbSize; ++i) {
		TrainSample &sample = dataBase.getSample(i);

		runModel(sample.input, transformation);

		size_t predicted_index = Activation::getMaxElementIndex(getOutput());

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

	if (showProgressbar) {
		bar.endPrint();
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
	return evaluateModel(dataBase, cancleOnError, true, transformation);
}

const global::Tensor &Model::getOutput() const {
	return network[network.size() - 1]->getOutput();
}

global::ValueType Model::getLoss(const global::Prediction &pre) {
	return network[network.size() - 1]->getLoss(pre);
}

size_t Model::outputSize() const {
	return network[network.size() - 1]->outputSize();
}

void Model::save(const std::string &file, const bool print) {
	std::ofstream outFile(file);

	ProgressBar bar(network.size(), "Saving Params to: " + file);

	for (size_t i = 0; i < network.size(); ++i) {
		std::vector<global::ValueType> params = network[i]->getParams();

		outFile << params.size() << " ";
		for (size_t j = 0; j < params.size(); ++j) {
			outFile << params[j] << " ";
		}
		outFile << std::endl;

		if (print) {
			bar.printBar();
			bar++;
		}
	}

	if (print) {
		bar.endPrint();
	}

	outFile.close();
}

void Model::load(const std::string &file, const bool print) {
	std::ifstream inFile(file);

	std::string line;
	int networkI = 0;

	ProgressBar bar(network.size(), "Loading Params from: " + file);

	while (std::getline(inFile, line)) {
		std::istringstream iss(line);

		size_t ParamSize;
		iss >> ParamSize;
		std::vector<global::ValueType> numbers(ParamSize);

		float num;
		for (size_t i = 0; i < ParamSize; ++i) {
			iss >> num;
			numbers[i] = num;
		}

		global::Tensor data({ParamSize});
		data = numbers;
		network[networkI]->setParams(data);
		networkI++;

		if (print) {
			bar.printBar();
			bar++;
		}
	}

	if (print) {
		bar.endPrint();
	}

	inFile.close();
}

global::Prediction Model::getPrediction() const {
	size_t max = 0;

	for (size_t i = 1; i < outputSize(); ++i) {
		if (getOutput().getValue(i) > getOutput().getValue(max)) {
			max = i;
		}
	}

	return global::Prediction(max, getOutput().getValue(max));
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

std::vector<global::ValueType> Model::getOut() const {
	std::vector<global::ValueType> temp(outputSize());
	getOutput().getData(temp);
	return temp;
}
} // namespace nn::model
