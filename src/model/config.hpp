#ifndef CONFIG
#define CONFIG

#include "activations.hpp"
#include <nlohmann/json.hpp>

namespace nn::model {

class ISubNetworkConfig {
  protected:
	int inputSize{0};
	int outputSize{0};

  public:
	virtual void fromJson(const nlohmann::json &j) = 0;
	virtual const std::string NNLable() const = 0;

	int getInputSize() const { return inputSize; }
	int getOutputSize() const { return outputSize; }

	virtual ~ISubNetworkConfig() = default;
};

struct DenseLayerConfig {
	int size;
	ActivationType activationType = ActivationType::None;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DenseLayerConfig, size, activationType)

class FNNConfig : public ISubNetworkConfig {
  public:
	FNNConfig(const nlohmann::json &j);
	~FNNConfig() = default;

	const std::string NNLable() const override { return "FNN"; }
	void fromJson(const nlohmann::json &j) override;

	std::vector<DenseLayerConfig> layersConfig;
};

class NetworkConfig {
  public:
	int inputSize() const;
	int outputSize() const;

	std::vector<std::shared_ptr<ISubNetworkConfig>> SubNetworksConfig;
	void fromJson(const nlohmann::json &j);
};

struct TrainingConfig {
	size_t batch_size;
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
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(VisualMode, state, mode);

struct VisualConfig {
	std::vector<VisualMode> modes;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(VisualConfig, modes);

class Config {
  public:
	Config(const std::string &config_filepath);
	VisualConfig visualConfig;
	TrainingConfig trainingConfig;
	NetworkConfig networkConfig;
};
} // namespace nn::model

#endif // CONFIG
