#include "config.hpp"
#include <cstddef>
#include <fstream>
#include <iostream>

namespace nn::model {

// ============================================================================
// CONFIG CLASS IMPLEMENTATION
// ============================================================================

Config::Config(const std::string &config_filepath) {
	std::ifstream ifs(config_filepath); // Open config file
	if (!ifs.is_open()) {               // Check if file is accessible
		std::cerr << "Error: Could not open config file: " << config_filepath
		          << std::endl;
		throw std::runtime_error("Failed to open config file: " +
		                         config_filepath);
	}

	nlohmann::json j;
	try {
		ifs >> j;         // Parse JSON from file
		initalizeJson(j); // Initialize internal configs from JSON
	} catch (const nlohmann::json::parse_error &e) {
		// Handle invalid JSON format
		std::cerr << "JSON parse error in file '" << config_filepath << "':\n"
		          << e.what() << "\n"
		          << "at byte " << e.byte << std::endl;
		throw;
	} catch (const nlohmann::json::exception &e) {
		// Handle general JSON issues (missing keys, invalid types, etc.)
		std::cerr << "JSON processing error in file '" << config_filepath << "':\n"
		          << e.what() << std::endl;
		throw;
	}
}

void Config::initalizeJson(const nlohmann::json &j) {
	// Load training config
	trainingConfig.fromJson(j.at("training config"));

	// Visual config is optional
	if (j.contains("visual config")) {
		visualConfig.fromJson(j.at("visual config"));
	}

	// Load network configuration (required)
	networkConfig.fromJson(j.at("network config"));
}

// ============================================================================
// NETWORK CONFIG IMPLEMENTATION
// ============================================================================

void NetworkConfig::fromJson(const nlohmann::json &j) {
	size_t prevS = 0; // Keeps track of previous output size for chaining

	for (auto &subNetworkConfig : j) {
		std::string type = subNetworkConfig.at("type");

		// Create appropriate sub-network config based on type
		if (type == fnn::FNN_LABLE) {
			SubNetworksConfig.push_back(
			    std::make_shared<fnn::FNNConfig>(subNetworkConfig, prevS));
		} else if (type == cnn::CNN_LABLE) {
			SubNetworksConfig.push_back(
			    std::make_shared<cnn::CNNConfig>(subNetworkConfig, prevS));
		}

		// Update previous size for next sub-network
		prevS = SubNetworksConfig[SubNetworksConfig.size() - 1]->getOutputSize();
	}
}

// ============================================================================
// FEEDFORWARD NETWORK CONFIG IMPLEMENTATION
// ============================================================================

namespace fnn {

FNNConfig::FNNConfig(const nlohmann::json &j, const size_t prevS) {
	inputShape.resize(1);

	// First network uses "input size", otherwise inherit from prev network
	if (prevS == 0) {
		inputShape[0] = j.at("input size");
	} else {
		inputShape[0] = prevS;
	}

	// Output size of the network
	outputSize = j.at("output size");

	// Load dense layers
	for (auto &layer_ : j.at("layers")) {
		layersConfig.push_back(DenseLayerConfig(layer_));
	}

	// Output activation function
	outputActivation = (ActivationType)j.at("output activation");
}

DenseLayerConfig::DenseLayerConfig(const nlohmann::json &j) {
	fromJson(j); // Delegate parsing to helper
}

void DenseLayerConfig::fromJson(const nlohmann::json &j) {
	size = j.at("size"); // Layer size (neurons)

	// Optional dropout rate
	if (j.contains("dropoutRate")) {
		dropoutRate = j.at("dropoutRate");
	}

	// Optional activation type
	if (j.contains("activationType")) {
		activationType = j.at("activationType");
	}
}

} // namespace fnn

// ============================================================================
// CONVOLUTIONAL NETWORK CONFIG IMPLEMENTATION
// ============================================================================

namespace cnn {

CNNConfig::CNNConfig(const nlohmann::json &j, const size_t) {
	// Input tensor shape (w, h, f, c)
	inputShape = j.at("input shape").get<std::vector<size_t>>();

	// Output activation type
	activation = j.at("output activation");

	// Optional filter shape
	if (j.contains("filter shape")) {
		filterShape = j.at("filter shape").get<std::vector<size_t>>();
	}

	// Compute output size of CNN layer
	outputSize = calculateOutputSize();
}

size_t CNNConfig::calculateOutputSize() const {
	// Simple valid convolution output size calculation
	return (inputShape[0] - filterShape[0] + 1) *
	       (inputShape[1] - filterShape[1] + 1) * filterShape[2];
}

} // namespace cnn

// ============================================================================
// NETWORKCONFIG UTILITY METHODS
// ============================================================================

std::vector<size_t> NetworkConfig::inputShape() const {
	// Input shape is taken from the first sub-network
	return SubNetworksConfig[0]->getInputShape();
}

size_t NetworkConfig::outputSize() const {
	// Output size is taken from the last sub-network
	return SubNetworksConfig[SubNetworksConfig.size() - 1]->getOutputSize();
}

// ============================================================================
// TRAINING CONFIG IMPLEMENTATION
// ============================================================================

void TrainingConfig::fromJson(const nlohmann::json &j) {
	// Required training parameters
	batchCount = j.at("batch count");
	batchSize = j.at("batch size");

	// Optional auto-save config
	if (j.contains("auto save")) {
		autoSave = j.at("auto save").get<AutoSave>();
	}

	// Optional evaluation config
	if (j.contains("auto evaluating")) {
		autoEvaluating = j.at("auto evaluating").get<AutoEvaluating>();
		// Default: evaluate every 1% of batches
		autoEvaluating.evaluateEvery = batchCount / 100;
	}

	// Optimizer configuration
	if (j.contains("optimizer")) {
		nlohmann::json optimizerJ = j.at("optimizer");
		optimizerType = optimizerJ.at("type");

		// Currently only constant optimizer supported
		if (optimizerType == "const") {
			optimizer = std::make_unique<ConstantOptimizerConfig>(optimizerJ);
		}
	}
}

void ConstantOptimizerConfig::fromJson(const nlohmann::json &j) {
	learningRate = j.at("lr"); // Learning rate
}

// ============================================================================
// VISUAL CONFIG IMPLEMENTATION
// ============================================================================

void VisualConfig::fromJson(const nlohmann::json &j) {
	enableVisuals = j.at("enable visual"); // Enable/disable visualization

	if (!enableVisuals) {
		return; // Skip further parsing if visuals are off
	}

	// Optional FPS display
	if (j.contains("show fps")) {
		showFps = j.at("show fps");
	}

	// Enable network structure visualization
	enableNetwrokVisual = j.at("enable network visual");

	// Optional visualization modes
	if (j.contains("modes")) {
		modes = j.at("modes");
	}
}

} // namespace nn::model
