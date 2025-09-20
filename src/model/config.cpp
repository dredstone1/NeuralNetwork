/**
 * @file config.cpp
 * @brief Implementation of configuration management for neural network models
 * 
 * This file implements the configuration system that reads and parses JSON
 * configuration files to set up neural network parameters, training options,
 * and visualization settings. It provides a centralized way to manage all
 * model parameters and ensures proper validation of configuration data.
 */

#include "config.hpp"
#include <cstddef>
#include <fstream>
#include <iostream>

namespace nn::model {

/**
 * @brief Constructs a Config object by reading from a JSON configuration file
 * 
 * Loads and parses a JSON configuration file that contains all the necessary
 * parameters for setting up a neural network model, including network architecture,
 * training parameters, and visualization options.
 * 
 * @param config_filepath Path to the JSON configuration file
 * @throws std::runtime_error If the configuration file cannot be opened
 * @throws nlohmann::json::parse_error If the JSON is malformed
 * @throws nlohmann::json::exception If required configuration fields are missing
 */
Config::Config(const std::string &config_filepath) {
	std::ifstream ifs(config_filepath);
	if (!ifs.is_open()) {
		std::cerr << "Error: Could not open config file: " << config_filepath
		          << std::endl;
		throw std::runtime_error("Failed to open config file: " +
		                         config_filepath);
	}

	nlohmann::json j;
	try {
		ifs >> j;

		initalizeJson(j);
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

/**
 * @brief Initializes configuration objects from parsed JSON data
 * 
 * Takes a parsed JSON object and extracts configuration data for different
 * components of the neural network system. Handles training configuration,
 * visual configuration, and network architecture configuration.
 * 
 * @param j Parsed JSON configuration object
 * @throws nlohmann::json::exception If required configuration sections are missing
 */
void Config::initalizeJson(const nlohmann::json &j) {
	// Parse training configuration (required)
	trainingConfig.fromJson(j.at("training config"));

	// Parse visual configuration (optional)
	if (j.contains("visual config")) {
		visualConfig.fromJson(j.at("visual config"));
	}

	// Parse network architecture configuration (required)
	networkConfig.fromJson(j.at("network config"));
}

/**
 * @brief Initializes network configuration from JSON data
 * 
 * Parses the network architecture configuration section and creates
 * appropriate sub-network configurations for FNN and CNN layers.
 * Validates network connectivity and parameter consistency.
 * 
 * @param j JSON object containing network configuration data
 * @throws nlohmann::json::exception If network configuration is invalid
 */
void NetworkConfig::fromJson(const nlohmann::json &j) {
	size_t prevS = 0;
	for (auto &subNetworkConfig : j) {
		std::string type = subNetworkConfig.at("type");

		if (type == fnn::FNN_LABLE) {
			SubNetworksConfig.push_back(
			    std::make_shared<fnn::FNNConfig>(subNetworkConfig, prevS));
		} else if (type == cnn::CNN_LABLE) {
			SubNetworksConfig.push_back(
			    std::make_shared<cnn::CNNConfig>(subNetworkConfig, prevS));
		}

		prevS = SubNetworksConfig[SubNetworksConfig.size()-1]->getOutputSize();
	}
}

namespace fnn {

FNNConfig::FNNConfig(const nlohmann::json &j, const size_t prevS) {
	inputShape.resize(1);
	if (prevS == 0) {
		inputShape[0] = j.at("input size");
	} else {
		inputShape[0] = prevS;
	}

	outputSize = j.at("output size");

	for (auto &layer_ : j.at("layers")) {
		layersConfig.push_back(DenseLayerConfig(layer_));
	}

	outputActivation = (ActivationType)j.at("output activation");
}

DenseLayerConfig::DenseLayerConfig(const nlohmann::json &j) {
	fromJson(j);
}

void DenseLayerConfig::fromJson(const nlohmann::json &j) {
	size = j.at("size");

	if (j.contains("dropoutRate")) {
		dropoutRate = j.at("dropoutRate");
	}

	if (j.contains("activationType")) {
		activationType = j.at("activationType");
	}
}

} // namespace fnn

namespace cnn {

CNNConfig::CNNConfig(const nlohmann::json &j, const size_t) {
	inputShape = j.at("input shape").get<std::vector<size_t>>();

	activation = j.at("output activation");

	if (j.contains("filter shape")) {
		filterShape = j.at("filter shape").get<std::vector<size_t>>();
	}

	outputSize = calculateOutputSize();
}

size_t CNNConfig::calculateOutputSize() const {
	return (inputShape[0] - filterShape[0] + 1) *
	       (inputShape[1] - filterShape[1] + 1) * filterShape[2];
}

} // namespace cnn

std::vector<size_t> NetworkConfig::inputShape() const {
	return SubNetworksConfig[0]->getInputShape();
}

size_t NetworkConfig::outputSize() const {
	return SubNetworksConfig[SubNetworksConfig.size() - 1]->getOutputSize();
}

void TrainingConfig::fromJson(const nlohmann::json &j) {
	batchCount = j.at("batch count");
	batchSize = j.at("batch size");

	if (j.contains("auto save")) {
		autoSave = j.at("auto save").get<AutoSave>();
	}

	if (j.contains("auto evaluating")) {
		autoEvaluating = j.at("auto evaluating").get<AutoEvaluating>();
		autoEvaluating.evaluateEvery = batchCount / 100;
	}

	if (j.contains("optimizer")) {
		nlohmann::json optimizerJ = j.at("optimizer");
		optimizerType = optimizerJ.at("type");

		if (optimizerType == "const") {
			optimizer = std::make_unique<ConstantOptimizerConfig>(optimizerJ);
		}
	}
}

void ConstantOptimizerConfig::fromJson(const nlohmann::json &j) {
	learningRate = j.at("lr");
}

void VisualConfig::fromJson(const nlohmann::json &j) {
	enableVisuals = j.at("enable visual");
	if (!enableVisuals) {
		return;
	}

	if (j.contains("show fps")) {
		showFps = j.at("show fps");
	}

	enableNetwrokVisual = j.at("enable network visual");

	if (j.contains("modes")) {
		modes = j.at("modes");
	}
}

} // namespace nn::model
