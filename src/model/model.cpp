#include "../networks/cnn/CNNetwork.hpp"
#include "../networks/fnn/FNNetwork.hpp"
#include "ProgressBar.hpp"
#include "config.hpp"
#include "dataBase.hpp"
#include "loss_function.hpp"
#include "tensor_gpu.hpp"
#include <fstream>
#include <iostream>
#include <model.hpp>
#include <random>
#include <vector>

namespace nn::model {

Model::Model(const std::string &config_filepath)
    : config(config_filepath),
      visual(config),
      learningRate(config.trainingConfig.getLearningRate()),
      output({config.networkConfig.outputSize()}) {
	initOptimizer();
	initModel();
	if (config.visualConfig.enableVisuals) {
		initVisual();
	}
}

void Model::initOptimizer() {
	const std::string &type = config.trainingConfig.getOptimizerType();

	if (type == "const") {
		auto *optConfig = dynamic_cast<ConstantOptimizerConfig *>(
		    config.trainingConfig.getOptimizer().get());
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
	const fnn::FNNConfig &sub_ = (fnn::FNNConfig &)(_config);
	std::shared_ptr<visualizer::fnn::FnnVisualier> visual_ = nullptr;

	if (shouldRenderNet()) {
		visual_ = std::make_shared<visualizer::fnn::FnnVisualier>(
		    visual.Vstate, width, sub_);
	}

	network.push_back(std::make_unique<fnn::FNNetwork>(sub_, true, visual_));
}

void Model::addCNN(const std::uint32_t width, ISubNetworkConfig &_config) {
	const cnn::CNNConfig &sub_ = (cnn::CNNConfig &)(_config);
	std::shared_ptr<visualizer::cnn::CnnVisualier> visual_ = nullptr;

	if (shouldRenderNet()) {
		visual_ = std::make_shared<visualizer::cnn::CnnVisualier>(
		    visual.Vstate, width, sub_);
	}

	network.push_back(std::make_unique<cnn::CNNetwork>(sub_, true, visual_));
}

bool Model::shouldRenderNet() const {
	return config.visualConfig.enableVisuals &&
	       config.visualConfig.enableNetwrokVisual;
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

void Model::Backward(global::Tensor &output, const global::ValueType weight) {
	global::Tensor *delta = &output;

	for (int i = static_cast<int>(network.size()) - 1; i >= 0; --i) {
		network[i]->backward(&delta, weight);
		delta = network[i]->getInput();
	}
}

global::ValueType Model::runBackPropagation(
    const Batch &batch, DataBase &db,
    const bool doBackward) {
	global::ValueType error = 0.0;

	if (batch.size() == 0) {
		return error;
	}

	resetNetworkGradient();

	Loss loss(config.trainingConfig.getLossType(),
	          db.samples.status.outputType);
	for (size_t i = 0; i < batch.size(); ++i) {
		TrainSample current_sample_ptr = db.getSample(batch.samples[i]);

		if (current_sample_ptr.out) {
			output = *current_sample_ptr.out;
		} else {
			output.zero();
			output.setValue(current_sample_ptr.index, 1);
		}

		visual.updatePrediction(output);
		runModel(current_sample_ptr.input);

		if (doBackward) {
			Backward(output, current_sample_ptr.weight);
		}

		error += loss.LossF(current_sample_ptr.index, current_sample_ptr.out,
		                    getOutput());
	}

	if (doBackward) {
		updateWeights(batch.size());
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

	std::cout << "Training Done!\n"
	          << "Training time: " << minutes << " minutes "
	          << seconds << " seconds ("
	          << duration_ms.count() << " ms)\n"
	          << "Final score: " << error << "\n";
}

void Model::resetTraining() {
	batchCounter = 0;

	if (config.visualConfig.enableVisuals) {
		visual.updateBatchCounter(batchCounter);
		visual.resetGraph();
	}
}

void Model::train(DataBase &dbT, DataBase &dbE) {
	try {
		trainModel(dbT, dbE);
	} catch (const std::exception &e) {
		std::cerr << "Training failed: " << e.what() << std::endl;
		throw;
	}
}

bool Model::autoEvaluating(
    const int i,
    DataBase &evaluateDataBase) {
	setEvaluating();
	if (config.trainingConfig.isAutoEvaluating() &&
	    i % config.trainingConfig.getAutoEvaluating().evaluateEvery == 0) {
		modelResult result = evaluateModel(evaluateDataBase, false, false);
		visual.updateEvaluate(
		    result.percentage,
		    i / config.trainingConfig.getAutoEvaluating().evaluateEvery);

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

void Model::generateBatches(DataBase &db, std::vector<Batch> &batches) {
	const size_t data_size = db.samples.status.dbSize;
	const size_t batch_size = config.trainingConfig.getBatchSize();

	// Shuffle indices
	std::vector<size_t> indices(data_size);
	std::iota(indices.begin(), indices.end(), 0);

	std::mt19937 rng{std::random_device{}()};
	std::shuffle(indices.begin(), indices.end(), rng);

	// Prepare batches
	batches.clear();
	batches.reserve((data_size + batch_size - 1) / batch_size);

	for (size_t start = 0; start < data_size; start += batch_size) {
		size_t current_size = std::min(batch_size, data_size - start);

		Batch &batch = batches.emplace_back(current_size);
		std::copy_n(indices.begin() + start, current_size, batch.samples.begin());
	}
}

Batch &Model::getBatch(DataBase &db, size_t &index, std::vector<Batch> &batches) {
	if (batches.empty() || index >= batches.size()) {
		generateBatches(db, batches);
		index = 0;
	}

	return batches.at(index++);
}

void Model::trainModel(DataBase &trainedDataBase, DataBase &evaluateDataBase) {
	ProgressBar bar(config.trainingConfig.getBatchCount(), TRAINING_HEADER);
	const auto start = std::chrono::high_resolution_clock::now();
	global::ValueType error = 0.0;

	std::vector<Batch> batches;
	generateBatches(trainedDataBase, batches);
	size_t currentBatch;

	visual.updateLearningRate(learningRate);
	setTraining();
	bar = batchCounter;

	for (; batchCounter < config.trainingConfig.getBatchCount() + 1; ++batchCounter) {
		++bar;
		bar.printBar();

		visual.updateBatchCounter(batchCounter);

		Batch &batch = getBatch(trainedDataBase, currentBatch, batches);
		error = runBackPropagation(batch, trainedDataBase, true);
		visual.updateLoss(error, batchCounter);

		autoSave(batchCounter);

		if (visual.exitTraining() ||
		    autoEvaluating(batchCounter, evaluateDataBase)) {
			break;
		}

		setTraining();

		visual.updateLearningRate(learningRate);
	}
	setNormal();
	bar.endPrint();
	printTrainingResult(start, error);
}

float Model::calculatePercentage(float currentSize, float totalSize) {
	if (totalSize == 0) {
		return 0.0f;
	}

	return 100.0f * currentSize / totalSize;
}

modelResult Model::evaluateModel(
    DataBase &dataBase, const bool cancleOnError, const bool showProgressbar) {
	if (dataBase.samples.status.outputType == OutType::Statistic) {
		return {};
	}

	modelResult result{0, 0, 0};

	ProgressBar bar(dataBase.DataBaseLength(), EVALUATING_HEADER);
	result.dbSize = 0;

	setEvaluating();

	for (size_t i = 0; i < dataBase.DataBaseLength(); ++i) {
		TrainSample sample = dataBase.getSample(i);

		result.dbSize += sample.weight;
		runModel(sample.input);

		size_t predicted_index = Activation::getMaxElementIndex(getOutput());

		if (showProgressbar) {
			++bar;
			bar.printBar();
		}

		if (predicted_index == sample.index) {
			result.currectPreSize += sample.weight;
		} else if (cancleOnError) {
			break;
		}
	}

	if (showProgressbar) {
		bar.endPrint();
	}

	result.percentage = calculatePercentage(result.currectPreSize,
	                                        result.dbSize);
	setNormal();
	return result;
}

const global::Tensor &Model::getOutput() const {
	return network[network.size() - 1]->getOutput();
}

size_t Model::outputSize() const {
	return network[network.size() - 1]->outputSize();
}

void Model::save(const std::string &file, const bool print) {
	std::ofstream outFile(file);

	ProgressBar bar(network.size(), SAVING_DATA_HEADER + file);

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

void Model::load(const std::string &file, bool print) {
	std::ifstream inFile(file);
	if (!inFile) {
		throw std::runtime_error("Error: Could not open file \"" + file + "\" for reading.");
	}

	std::string line;
	int networkI = 0;
	ProgressBar bar(network.size(), LOADING_DATA_HEADER + file);

	while (std::getline(inFile, line)) {
		if (networkI >= static_cast<int>(network.size())) {
			throw std::runtime_error(
			    "Error in file \"" + file + "\": too many sub-networks. "
			                                "Expected " +
			    std::to_string(network.size()) +
			    " but found more (at line " + std::to_string(networkI + 1) + ").");
		}

		std::istringstream iss(line);
		size_t ParamSize = 0;
		if (!(iss >> ParamSize)) {
			throw std::runtime_error(
			    "Error in file \"" + file + "\": invalid parameter size "
			                                "at line " +
			    std::to_string(networkI + 1) + ".");
		}

		size_t expectedParams = network[networkI]->getParamCount();
		if (ParamSize != expectedParams) {
			throw std::runtime_error(
			    "Error in file \"" + file + "\": parameter count mismatch "
			                                "in sub-network " +
			    std::to_string(networkI + 1) +
			    ". Expected " + std::to_string(expectedParams) +
			    ", got " + std::to_string(ParamSize) + ".");
		}

		std::vector<global::ValueType> numbers(ParamSize);
		for (size_t i = 0; i < ParamSize; ++i) {
			float num = 0.0f;
			if (!(iss >> num)) {
				throw std::runtime_error(
				    "Error in file \"" + file + "\": invalid parameter value "
				                                "at line " +
				    std::to_string(networkI + 1) +
				    ", parameter " + std::to_string(i + 1) + ".");
			}
			numbers[i] = num;
		}

		global::Tensor data({ParamSize});
		data = numbers;
		network[networkI]->setParams(data);
		++networkI;

		if (print) {
			bar.printBar();
			++bar;
		}
	}

	if (networkI < static_cast<int>(network.size())) {
		throw std::runtime_error(
		    "Error in file \"" + file + "\": file ended prematurely. "
		                                "Expected " +
		    std::to_string(network.size()) +
		    " sub-networks, but only " + std::to_string(networkI) + " were provided.");
	}

	if (print) {
		bar.endPrint();
	}
}

Prediction Model::getPrediction() const {
	size_t max = 0;

	for (size_t i = 1; i < outputSize(); ++i) {
		if (getOutput().getValue(i) > getOutput().getValue(max)) {
			max = i;
		}
	}

	return Prediction(max, getOutput().getValue(max));
}

void Model::setTraining(const bool state) {
	for (auto &sub : network) {
		sub->setTraining(state);
	}
}

void Model::setTraining() {
	visual.updateAlgorithmMode(visualizer::AlgorithmMode::Training);
	setTraining(true);
}

void Model::setNormal() {
	visual.updateAlgorithmMode(visualizer::AlgorithmMode::Normal);
	setTraining(false);
}

void Model::setEvaluating() {
	visual.updateAlgorithmMode(visualizer::AlgorithmMode::Evaluating);
	setTraining(false);
}

std::vector<global::ValueType> Model::getOut() const {
	std::vector<global::ValueType> temp(outputSize());
	getOutput().getData(temp);
	return temp;
}

} // namespace nn::model
