#ifndef CONFIG
#define CONFIG

#include "activations.hpp"
#include <Globals.hpp>
#include <memory>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>

namespace nn::model {

class ISerializable {
  public:
	virtual void fromJson(const nlohmann::json &j) = 0;
	virtual ~ISerializable() = default;
};

class ISubNetworkConfig {
  public:
	virtual void fromJson(const nlohmann::json &j) = 0;
	virtual const std::string NNLable() const = 0;
};

struct DenseLayerConfig {
	int size;
	ActivationType activationType;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    DenseLayerConfig,
    size,
    activationType)

class FNNConfig : public ISubNetworkConfig {
  private:
	void fromJson(const nlohmann::json &j) override;
	friend class Config;

  public:
	FNNConfig(const nlohmann::json &j);

	const std::string NNLable() const override { return "FNN"; }

	std::vector<DenseLayerConfig> layersConfig;
	int inputSize;
	int outputSize;
};

class NetworkConfig : public ISerializable {
  public:
	std::vector<std::unique_ptr<ISubNetworkConfig>> SubNetworksConfig;
	void fromJson(const nlohmann::json &j) override;
};

struct TrainingConfig {
	int batch_size;
	int batch_count;
	std::string db_filename;
	global::ValueType lr_init_value = 0.001;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    TrainingConfig,
    batch_size,
    batch_count,
    db_filename,
    lr_init_value);

struct VisualMode {
	std::string state;
	bool mode = true;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    VisualMode,
    state,
    mode);

struct VisualConfig {
	std::vector<VisualMode> modes;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    VisualConfig,
    modes);

class Config {
  public:
	Config(const std::string &config_filepath);
	VisualConfig visualConfig;
	TrainingConfig trainingConfig;
	NetworkConfig networkConfig;
};
} // namespace nn::model

#endif // CONFIG
