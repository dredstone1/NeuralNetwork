#ifndef CONFIG
#define CONFIG

#include <Globals.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace nn::model {

class ISerializable {
  public:
	virtual void fromJson(const nlohmann::json &j) = 0;
	virtual ~ISerializable() = default;
};

// struct TrainingConfig {
// 	int batch_size;
// 	int batch_count;
// 	std::string db_filename;
// 	global::ValueType lr_init_value = 0.001;
// };
// NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
//     TrainingConfig,
//     batch_size,
//     batch_count,
//     db_filename,
//     lr_init_value);
//
// struct VisualMode {
// 	std::string state;
// 	bool mode = true;
// };
// NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
//     VisualMode,
//     state,
//     mode);
//
// struct VisualizerConfig {
// 	std::vector<VisualMode> modes;
// };
// NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
//     VisualizerConfig,
//     modes);
//
// struct ConfigData {
// 	// NetworkConfig network_config;
// 	TrainingConfig training_config;
// 	VisualizerConfig visualizer_config;
// };
// NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
//     ConfigData,
//     // network_config,
//     training_config,
//     visualizer_config);

class Config {
  public:
	Config(const std::string &config_filepath);
	// ConfigData config_data;
};
} // namespace nn::model

#endif // CONFIG
