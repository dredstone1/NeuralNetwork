/**
 * @file model.cpp
 * @brief Implementation of the Model class for neural network training and evaluation
 * 
 * This file contains the core implementation of the Model class, which manages
 * the complete lifecycle of neural network models including construction,
 * training, evaluation, and visualization. It coordinates between different
 * network types (CNN, FNN) and handles optimization, data management, and
 * progress tracking.
 */

#include "../networks/cnn/CNNetwork.hpp"
#include "../networks/fnn/FNNetwork.hpp"
#include "ProgressBar.hpp"
#include <fstream>
#include <iostream>
#include <model.hpp>
#include <random>
#include <stdexcept>

namespace nn::model {

/**
 * @brief Constructs a Model from a configuration file
 * 
 * Initializes a complete neural network model by reading configuration from
 * a JSON file. This constructor sets up the optimizer, network architecture,
 * and visualization components based on the provided configuration.
 * 
 * @param config_filepath Path to the JSON configuration file
 * @throws std::runtime_error If configuration file cannot be read or parsed
 * @throws std::invalid_argument If configuration contains invalid parameters
 */
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

/**
 * @brief Initializes the optimization algorithm based on configuration
 * 
 * Creates and configures the appropriate optimizer (e.g., Constant, Adam, SGD)
 * based on the optimizer type specified in the training configuration.
 * The optimizer is responsible for updating network parameters during training.
 * 
 * @throws std::runtime_error If unknown optimizer type is specified
 */
void Model::initOptimizer() {
	const std::string &type = config.trainingConfig.getOptimizerType();

	if (type == "const") {
		auto *optConfig = dynamic_cast<ConstantOptimizerConfig *>(
		    config.trainingConfig.getOptimizer().get());
		optimizer = std::make_unique<ConstantOptimizer>(*optConfig);
	}
}

/**
 * @brief Initializes the visualization system for the model
 * 
 * Sets up the visual components for real-time monitoring of training progress
 * and network visualization. This includes graphs for loss tracking and
 * visual representation of network layers if enabled in configuration.
 * 
 * @note Only initializes if visualization is enabled in config
 */
void Model::initVisual() {
	visual.start();

	// Connect network visual components if network visualization is enabled
	if (!config.visualConfig.enableNetwrokVisual) {
		return;
	}

	// Link each sub-network's visual component to the main visualizer
	for (size_t i = 0; i < config.networkConfig.SubNetworksConfig.size(); ++i) {
		visual.addVisualSubNetwork(network[i]->getVisual());
		network[i]->getVisual()->setVstate(visual.Vstate);
	}
}

/**
 * @brief Calculates the width allocation for each sub-network in visualization
 * 
 * Determines how much horizontal space each sub-network should occupy in the
 * visual representation by dividing the total available width by the number
 * of sub-networks.
 * 
 * @return Width in pixels for each sub-network, or 0 if no sub-networks exist
 */
std::uint32_t Model::calculateSubNetWidth() const {
	const auto count = config.networkConfig.SubNetworksConfig.size();
	if (count == 0) {
		return 0;
	}
	return visualizer::SUB_NETWORKS_WIDTH / static_cast<std::uint32_t>(count);
}

/**
 * @brief Initializes the neural network architecture
 * 
 * Constructs the complete neural network by creating and connecting all
 * sub-networks (CNNs, FNNs) according to the configuration. Also calculates
 * the total number of trainable parameters and prints model information.
 * 
 * @throws std::runtime_error If network configuration is invalid
 * @throws std::bad_alloc If insufficient memory for network creation
 */
void Model::initModel() {
	const std::uint32_t WIDTH = calculateSubNetWidth();
	size_t param_amount = 0;

	// Create and configure each sub-network according to its type
	for (size_t i = 0; i < config.networkConfig.SubNetworksConfig.size(); ++i) {
		ISubNetworkConfig &_config = *config.networkConfig.SubNetworksConfig[i];

		// Instantiate the appropriate network type based on configuration
		if (_config.NNLable() == fnn::FNN_LABLE) {
			addFNN(WIDTH, _config);
		} else if (_config.NNLable() == cnn::CNN_LABLE) {
			addCNN(WIDTH, _config);
		}

		param_amount += network[i]->getParamCount();
	}

	// Print model summary information
	std::cout << "initialize model - "
	          << param_amount << " parameters, "
	          << config.networkConfig.SubNetworksConfig.size() << " sub networks"
	          << std::endl;
}

/**
 * @brief Adds a Fully Connected Network (FNN) to the model
 * 
 * Creates and configures a new FNN sub-network with the specified parameters.
 * Optionally creates a visual component for real-time network visualization.
 * 
 * @param width Visual width allocation for this sub-network in pixels
 * @param _config Configuration object containing FNN-specific parameters
 * @throws std::bad_cast If _config is not a valid FNNConfig
 */
void Model::addFNN(const std::uint32_t width, ISubNetworkConfig &_config) {
	fnn::FNNConfig &sub_ = (fnn::FNNConfig &)(_config);
	std::shared_ptr<visualizer::fnn::FnnVisualier> visual_ = nullptr;

	// Create visualizer component if network visualization is enabled
	if (shouldRenderNet()) {
		visual_ = std::make_shared<visualizer::fnn::FnnVisualier>(
		    visual.Vstate, width, sub_);
	}

	network.push_back(std::make_unique<fnn::FNNetwork>(sub_, true, visual_));
}

/**
 * @brief Adds a Convolutional Neural Network (CNN) to the model
 * 
 * Creates and configures a new CNN sub-network with the specified parameters.
 * Optionally creates a visual component for real-time network visualization.
 * 
 * @param width Visual width allocation for this sub-network in pixels
 * @param _config Configuration object containing CNN-specific parameters
 * @throws std::bad_cast If _config is not a valid CNNConfig
 */
void Model::addCNN(const std::uint32_t width, ISubNetworkConfig &_config) {
	cnn::CNNConfig &sub_ = (cnn::CNNConfig &)(_config);
	std::shared_ptr<visualizer::cnn::CnnVisualier> visual_ = nullptr;

	// Create visualizer component if network visualization is enabled
	if (shouldRenderNet()) {
		visual_ = std::make_shared<visualizer::cnn::CnnVisualier>(
		    visual.Vstate, width, sub_);
	}

	network.push_back(std::make_unique<cnn::CNNetwork>(sub_, true, visual_));
}

/**
 * @brief Determines if network visualization should be rendered
 * 
 * Checks if both general visualization and network-specific visualization
 * are enabled in the configuration.
 * 
 * @return true if network visualization should be rendered, false otherwise
 */
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
	global::Tensor output({outputSize()});
	for (size_t i = 0; i < batch.size(); ++i) {
		TrainSample current_sample_ptr = db.getSample(batch.samples.at(i));
		visual.updatePrediction(current_sample_ptr.pre);

		runModel(current_sample_ptr.input);

		if (doBackward) {
			output.zero();
			output.setValue(current_sample_ptr.pre.index, 1);
			Backward(output, current_sample_ptr.weight);
		}

		error += getLoss(current_sample_ptr.pre);
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
		visual.updateLost(error, batchCounter);

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

		if (predicted_index == sample.pre.index) {
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

global::ValueType Model::getLoss(const global::Prediction &pre) {
	return network[network.size() - 1]->getLoss(pre);
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

global::Prediction Model::getPrediction() const {
	size_t max = 0;

	for (size_t i = 1; i < outputSize(); ++i) {
		if (getOutput().getValue(i) > getOutput().getValue(max)) {
			max = i;
		}
	}

	return global::Prediction(max, getOutput().getValue(max));
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
