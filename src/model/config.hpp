#ifndef CONFIG
#define CONFIG

#include "activations.hpp"
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
	global::ValueType learningRate{0.1};

  public:
	ConstantOptimizerConfig(const nlohmann::json &j) { fromJson(j); }
	void fromJson(const nlohmann::json &j) override;
	global::ValueType getLearningRate() const override { return learningRate; }

	~ConstantOptimizerConfig() = default;
};

class ISubNetworkConfig {
  protected:
	std::vector<size_t> inputShape;
	size_t outputSize{0};

  public:
	virtual void fromJson(const nlohmann::json &j) = 0;
	virtual const std::string NNLable() const = 0;

	std::vector<size_t> getInputShape() const { return inputShape; }
	size_t getOutputSize() const { return outputSize; }

	virtual ~ISubNetworkConfig() = default;
};

namespace fnn {
class DenseLayerConfig {
  public:
	DenseLayerConfig(const nlohmann::json &j);
	size_t size;
	float dropoutRate{0};
	ActivationType activationType = ActivationType::None;
	void fromJson(const nlohmann::json &j);
};

const std::string FNN_LABLE = "FNN";
class FNNConfig : public ISubNetworkConfig {
  public:
	FNNConfig(const nlohmann::json &j);
	~FNNConfig() = default;

	const std::string NNLable() const override { return FNN_LABLE; }
	void fromJson(const nlohmann::json &j) override;

	std::vector<DenseLayerConfig> layersConfig;
	ActivationType outputActivation;
};
} // namespace fnn

namespace cnn {
const std::string CNN_LABLE = "CNN";
class CNNConfig : public ISubNetworkConfig {
  public:
	CNNConfig(const nlohmann::json &j);
	~CNNConfig() = default;

	const std::string NNLable() const override { return CNN_LABLE; }
	void fromJson(const nlohmann::json &j) override;

	ActivationType activation;
	std::vector<size_t> filterShape{3, 3, 1, 1}; // {w, h, f, c}
};
} // namespace cnn

class NetworkConfig {
  public:
	std::vector<size_t> inputShape() const;
	size_t outputSize() const;

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
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(AutoEvaluating, dataBaseFilename);

class TrainingConfig {
  private:
	AutoSave autoSave;
	AutoEvaluating autoEvaluating;

	size_t batchSize;
	size_t batchCount;

	std::unique_ptr<IOptimizerConfig> optimizer;
	std::string optimizerType;

  public:
	TrainingConfig() {}
	~TrainingConfig() = default;

	void fromJson(const nlohmann::json &j);

	bool isAutoSave() const { return autoSave.saveEvery > 0; }
	const AutoSave &getAutoSave() const { return autoSave; }

	bool isAutoEvaluating() const { return autoEvaluating.evaluateEvery > 0; }
	const AutoEvaluating &getAutoEvaluating() const { return autoEvaluating; }

	size_t getBatchCount() const { return batchCount; }
	size_t getBatchSize() const { return batchSize; }

	global::ValueType getLearningRate() const {
		return optimizer->getLearningRate();
	}

	const std::unique_ptr<IOptimizerConfig> &getOptimizer() const {
		return optimizer;
	}
	const std::string &getOptimizerType() const { return optimizerType; }
};

struct VisualMode {
	std::string state;
	bool mode = true;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(VisualMode, state, mode);

class VisualConfig {
  public:
	VisualConfig() {}
	~VisualConfig() = default;

	bool enableVisuals{false};
	bool enableNetwrokVisual{false};
	std::vector<VisualMode> modes;
	void fromJson(const nlohmann::json &j);
};

class Config {
  public:
	Config(const std::string &config_filepath);
	VisualConfig visualConfig;
	TrainingConfig trainingConfig;
	NetworkConfig networkConfig;
};
} // namespace nn::model

#endif // CONFIG
