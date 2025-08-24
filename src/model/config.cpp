#include "config.hpp"
#include <fstream>
#include <iostream>

namespace nn::model {
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

void Config::initalizeJson(const nlohmann::json &j) {
	trainingConfig.fromJson(j.at("training config"));

	if (j.contains("visual config")) {
		visualConfig.fromJson(j.at("visual config"));
	}

	networkConfig.fromJson(j.at("network config"));
}

void NetworkConfig::fromJson(const nlohmann::json &j) {
	for (auto &subNetworkConfig : j) {
		std::string type = subNetworkConfig.at("type");

		if (type == fnn::FNN_LABLE) {
			SubNetworksConfig.push_back(std::make_shared<fnn::FNNConfig>(subNetworkConfig));
		} else if (type == cnn::CNN_LABLE) {
			SubNetworksConfig.push_back(std::make_shared<cnn::CNNConfig>(subNetworkConfig));
		}
	}
}

namespace fnn {

FNNConfig::FNNConfig(const nlohmann::json &j) {
	fromJson(j);
}

void FNNConfig::fromJson(const nlohmann::json &j) {
	inputShape.resize(1);
	inputShape[0] = j.at("input size");

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
	for (auto i : j.at("input shape")) {
		inputShape.push_back(i);
	}

	outputSize = j.at("output size");

	activation = (ActivationType)j.at("output activation");

	if (j.contains("filter shape")) {
		size_t k = 0;
		for (const auto &val : j.at("filter shape")) {
			if (k < filterShape.size()) {
				filterShape[k++] = val.get<size_t>();
			}
		}
	}
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
