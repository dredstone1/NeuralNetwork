#ifndef MODEL
#define MODEL

#include "dataBase.hpp"
#include "../visualizer/VisualizerController.hpp"
#include "Globals.hpp"
#include "INetwork.hpp"

namespace nn::model {
constexpr int BAR_WIDTH = 100;
constexpr int SECONDS_IN_MINUTE = 60;
struct learningRateParams {
	global::ValueType currentLearningRate;
	learningRateParams(const global::ValueType initLearningRate) : currentLearningRate(initLearningRate) {}
};

class Model {
  private:
	std::vector<std::unique_ptr<INetwork>> network;
	visualizer::VisualManager visual;

	// const ConfigData &config;
	learningRateParams learningRate;
	DataBase dataBase;

	void Forword(const global::ParamMetrix &input, const int modelIndex);
	void Backward(const global::ParamMetrix &output);
	void update_weights(const int batch_size);
	void resetNetworkGradient();
	void initModel();
	global::ValueType getLost(const global::ParamMetrix &output);

	global::ValueType run_back_propagation(const Batch &batch);

	void loadConfig(const std::string &config_filepath);

  public:
	Model(const std::string &config_filepath);
	~Model() = default;

	void runModel(const global::ParamMetrix &input);
	void train();

	void reset();
	void updateWeights(const global::ValueType learningRate);

	int outputSize();
	int inputSize();
	const global::ParamMetrix &getOutput() const;
};
} // namespace nn::model

#endif // MODEL
