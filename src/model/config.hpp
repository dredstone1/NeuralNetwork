#ifndef CONFIG
#define CONFIG

#include "activations.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace nn {
enum class DecayType {
	Constant = 0,
	StepDecay = 1,
	ExponentialDecay = 2,
};

struct LayerConfig {
	int size;
	Global::ValueType weights_init_value = -1;
	activation AT = activation::leaky_relu_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LayerConfig, size, AT, weights_init_value);

struct NetworkConfig {
	int input_size;
	int output_size;
	Global::ValueType output_init_value = -1;
	std::vector<LayerConfig> layers_config;
	size_t hidden_layer_count() const { return layers_config.size(); }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NetworkConfig, input_size, output_size, output_init_value, layers_config);

struct LrConfig {
	DecayType decay_type = DecayType::Constant;
	Global::ValueType lr_init_value = 0.001;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    LrConfig,
    decay_type,
    lr_init_value);

struct TrainingConfig {
	LrConfig lr_config;
	int batch_size;
	int batch_count;
	std::string db_filename;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    TrainingConfig,
    lr_config,
    batch_size,
    batch_count,
    db_filename);

struct VisualMode {
	std::string state;
	bool mode = true;
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
} // namespace nn

#endif // CONFIG
