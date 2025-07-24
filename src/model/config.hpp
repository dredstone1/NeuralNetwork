#ifndef CONFIG
#define CONFIG

#include "Globals.hpp"
#include "activations.hpp"
#include <memory>
#include <nlohmann/json.hpp>

namespace nn::model {
class IOptimizerConfig {
  public:
	virtual void fromJson(const nlohmann::json &j) = 0;
	virtual global::ValueType getLearningRate() const = 0;

	virtual ~IOptimizerConfig() = default;
};

class ConstantOptimizerConfig : public IOptimizerConfig {
  private:
	global::ValueType learningRate{0.001};

  public:
	ConstantOptimizerConfig(const nlohmann::json &j) { fromJson(j); }
	void fromJson(const nlohmann::json &j) override;
	global::ValueType getLearningRate() const override { return learningRate; }

	~ConstantOptimizerConfig() = default;
};

class AdamOptimizerConfig : public IOptimizerConfig {
  private:
	global::ValueType learningRate{0.001};
	global::ValueType beta1{0.9};
	global::ValueType beta2{0.999};
	global::ValueType epsilon{1e-8};

  public:
	AdamOptimizerConfig() = default;
	AdamOptimizerConfig(const nlohmann::json &j) { fromJson(j); }

	void fromJson(const nlohmann::json &j) override {
		if (j.contains("lr"))       learningRate = j["lr"].get<global::ValueType>();
		if (j.contains("beta1"))    beta1        = j["beta1"].get<global::ValueType>();
		if (j.contains("beta2"))    beta2        = j["beta2"].get<global::ValueType>();
		if (j.contains("epsilon"))  epsilon      = j["epsilon"].get<global::ValueType>();
	}

	global::ValueType getLearningRate() const override { return learningRate; }

	// Accessors for internal Adam parameters
	global::ValueType getBeta1()   const { return beta1; }
	global::ValueType getBeta2()   const { return beta2; }
	global::ValueType getEpsilon() const { return epsilon; }

	~AdamOptimizerConfig() = default;
};


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

	std::unique_ptr<IOptimizerConfig> optimizer;
	std::string optimizerType;

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

	global::ValueType getLearningRate() const { return optimizer->getLearningRate(); }

	const std::unique_ptr<IOptimizerConfig> &getOptimizer() const { return optimizer; }
	const std::string &getOptimizerType() const { return optimizerType; }
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
