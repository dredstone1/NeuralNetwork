#ifndef MODEL
#define MODEL

#include "../visualizer/VisualizerController.hpp"
#include "Globals.hpp"
#include "INetwork.hpp"
#include "dataBase.hpp"

namespace nn::model {
constexpr int BAR_WIDTH = 100;
constexpr int SECONDS_IN_MINUTE = 60;

class Model {
  private:
	std::vector<std::unique_ptr<INetwork>> network;
	std::unique_ptr<visualizer::VisualManager> visual;

	const Config config;
	global::ValueType learningRate;
	DataBase dataBase;

	void Forword(const global::ParamMetrix &input, const int modelIndex);
	void Backward(const global::ParamMetrix &output);
	void update_weights(const int batch_size);
	void resetNetworkGradient();
	void initModel();
	global::ValueType getLost(const global::ParamMetrix &output);

	global::ValueType run_back_propagation(const Batch &batch);

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
