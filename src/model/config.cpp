#include "config.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <vector>

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

		visualConfig = j.at("visual config").get<VisualConfig>();
		trainingConfig = j.at("training config").get<TrainingConfig>();

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
		if (type == "FNN") {
			SubNetworksConfig.push_back(std::make_shared<FNNConfig>(subNetworkConfig));
		}
	}
}

FNNConfig::FNNConfig(const nlohmann::json &j) {
	inputSize = j.at("input size");
	outputSize = j.at("output size");

	layersConfig = j.at("layers").get<std::vector<DenseLayerConfig>>();
}
} // namespace nn::model
