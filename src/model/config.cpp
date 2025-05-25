#include "config.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

Config::Config(const string &config_filepath) {
	ifstream ifs(config_filepath);
	if (!ifs.is_open()) {
		cerr << "Error: Could not open config file: " << config_filepath << endl;
		throw runtime_error("Failed to open config file: " + config_filepath);
	}

	nlohmann::json j;
	try {
		ifs >> j;
		this->config_data = j.get<ConfigData>();

	} catch (const nlohmann::json::parse_error &e) {
		cerr << "JSON parse error in file '" << config_filepath << "':\n"
		     << e.what() << "\n"
		     << "at byte " << e.byte << endl;
		throw;
	} catch (const nlohmann::json::exception &e) {
		cerr << "JSON processing error in file '" << config_filepath << "':\n"
		     << e.what() << endl;
		throw;
	}
}
