#ifndef MODEL
#define MODEL

#include "../visualizer/VisualizerController.hpp"
#include "INetwork.hpp"
#include "config.hpp"

namespace nn {
namespace training {
class BackPropagation;
class Trainer;
} // namespace training

namespace model {
struct learningRateParams {
	global::ValueType currentLearningRate;
	learningRateParams(const global::ValueType initLearningRate) : currentLearningRate(initLearningRate) {}
};

class Model {
  private:
	std::vector<std::unique_ptr<INetwork>> network;
	visualizer::VisualManager visual;

	const ConfigData &config;
	learningRateParams learningRate;

	void runModel(const global::ParamMetrix &input, const int modelIndex);

  public:
	Model(Config &_config);
	~Model() = default;

	void runModel(const global::ParamMetrix &input);
	void train();

	void reset();
	void updateWeights(const global::ValueType learningRate);

	int outputSize();
	const global::ParamMetrix &getOutput() const;
};
} // namespace model
} // namespace nn

#endif // MODEL
