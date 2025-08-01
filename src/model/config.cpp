#include "config.hpp"
#include "activations.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json_fwd.hpp>

namespace nn::model {
Config::Config(const std::string &config_filepath) {
	std::ifstream ifs(config_filepath);
	if (!ifs.is_open()) {
		std::cerr << "Error: Could not open config file: " << config_filepath << std::endl;
		throw std::runtime_error("Failed to open config file: " + config_filepath);
	}

	nlohmann::json j;
	try {
		ifs >> j;

		trainingConfig.fromJson(j.at("training config"));

		visualConfig = j.at("visual config").get<VisualConfig>();

		networkConfig.fromJson(j.at("network config"));
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

void NetworkConfig::fromJson(const nlohmann::json &j) {
	for (auto &subNetworkConfig : j) {
		std::string type = subNetworkConfig.at("type");
		if (type == fnn::FNN_LABLE) {
			SubNetworksConfig.push_back(std::make_shared<fnn::FNNConfig>(subNetworkConfig));
		}
		if (type == cnn::CNN_LABLE) {
			SubNetworksConfig.push_back(std::make_shared<cnn::CNNConfig>(subNetworkConfig));
		}
	}
}

namespace fnn {
FNNConfig::FNNConfig(const nlohmann::json &j) {
	fromJson(j);
}

void FNNConfig::fromJson(const nlohmann::json &j) {
	inputSize = j.at("input size");
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
CNNConfig::CNNConfig(const nlohmann::json &j) {
	fromJson(j);
}

void CNNConfig::fromJson(const nlohmann::json &j) {
	inputSize = j.at("input size");
	outputSize = j.at("output size");

	outputActivation = (ActivationType)j.at("output activation");
}
} // namespace cnn

int NetworkConfig::inputSize() const {
	return SubNetworksConfig[0]->getInputSize();
}

int NetworkConfig::outputSize() const {
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
} // namespace nn::model
