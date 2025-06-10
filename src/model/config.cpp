#include "config.hpp"
#include <fstream>
#include <iostream>

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
		config_data = j.get<ConfigData>();
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
} // namespace nn::model::config
