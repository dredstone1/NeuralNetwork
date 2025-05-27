#ifndef CONFIG_HPP
#define CONFIG_HPP

#include "activations.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace std;

struct LayerConfig {
	int size;
	double weights_init_value;
	ActivationFunctions::activations activation;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LayerConfig, size, activation, weights_init_value);

struct NetworkConfig {
	int input_size;
	int output_size;
	double output_init_value;
	vector<LayerConfig> layers_config;
	int hidden_layer_count() const { return layers_config.size(); }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NetworkConfig, input_size, output_size, output_init_value, layers_config);

struct TrainingConfig {
	double learning_rate;
	int batch_size;
	int batch_count;
	string db_filename;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TrainingConfig, learning_rate, batch_size, batch_count, db_filename)

struct VisualMode {
	string state;
	bool mode;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(VisualMode, state, mode);

struct VisualizerConfig {
	vector<VisualMode> modes;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(VisualizerConfig, modes);

struct ConfigData {
	NetworkConfig network_config;
	TrainingConfig training_config;
	VisualizerConfig visualizer_config;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ConfigData, network_config, training_config, visualizer_config);

class Config {
  public:
	Config(const string &config_filepath);
	ConfigData config_data;

  private:
};

#endif // CONFIG
