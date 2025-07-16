#ifndef CONFIG
#define CONFIG

#include "activations.hpp"
#include <memory>
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
	ActivationType outputActivation;
};

class NetworkConfig {
  public:
	int inputSize() const;
	int outputSize() const;

	std::vector<std::shared_ptr<ISubNetworkConfig>> SubNetworksConfig;
	void fromJson(const nlohmann::json &j);
};

struct AutoSave {
	int saveEvery{-1};
	std::string dataFilenameAutoSave{"model.txt"};
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(AutoSave, saveEvery, dataFilenameAutoSave);

struct AutoEvaluating {
	int evaluateEvery{-1};
	std::string dataBaseFilename{"dataBase"};
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(AutoEvaluating, evaluateEvery, dataBaseFilename);

class TrainingConfig {
  private:
	AutoSave autoSave;
	AutoEvaluating autoEvaluating;

	size_t batchSize;
	int batchCount;

	global::ValueType learningRate = 0.001;

  public:
	TrainingConfig() {}
	~TrainingConfig() = default;

	void fromJson(const nlohmann::json &j);

	bool isAutoSave() const { return (autoSave.saveEvery > 0); }
	const AutoSave &getAutoSave() const { return autoSave; }

	bool isAutoEvaluating() const { return (autoEvaluating.evaluateEvery > 0); }
	const AutoEvaluating &getAutoEvaluating() const { return autoEvaluating; }

	int getBatchCount() const { return batchCount; }
	size_t getBatchSize() const { return batchSize; }

	global::ValueType getLearningRate() const { return learningRate; }
};

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
