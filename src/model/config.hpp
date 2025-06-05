#ifndef CONFIG
#define CONFIG

#include "Globals.hpp"
#include "activations.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

struct LayerConfig {
	int size;
    Global::ValueType weights_init_value;
	activation AT;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LayerConfig, size, AT, weights_init_value);

struct NetworkConfig {
	int input_size;
	int output_size;
    Global::ValueType output_init_value;
	std::vector<LayerConfig> layers_config;
	int hidden_layer_count() const { return layers_config.size(); }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NetworkConfig, input_size, output_size, output_init_value, layers_config);

struct TrainingConfig {
    Global::ValueType learning_rate;
	int batch_size;
	int batch_count;
	std::string db_filename;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TrainingConfig, learning_rate, batch_size, batch_count, db_filename)

struct VisualMode {
	std::string state;
	bool mode;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(VisualMode, state, mode);

struct VisualizerConfig {
	std::vector<VisualMode> modes;
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
	Config(const std::string &config_filepath);
	ConfigData config_data;
};

#endif // CONFIG
