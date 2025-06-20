#ifndef MODEL
#define MODEL

#include "../visualizer/VisualizerController.hpp"
#include "INetwork.hpp"

namespace nn {
namespace training {
class BackPropagation;
class Trainer;
} // namespace training

namespace model {
class Model {
  private:
	std::vector<std::unique_ptr<INetwork>> network;
	visualizer::VisualManager visual;

	void runModel(const global::ParamMetrix &input, const int modelIndex);

	friend class training::BackPropagation;
	friend class training::Trainer;

  public:
	Model(Config &_config);
	~Model() = default;

	void runModel(const global::ParamMetrix &input);
	void reset();
	void updateWeights(const global::ValueType learningRate);

	int outputSize();
	const global::ParamMetrix &getOutput() const;
};
} // namespace model
} // namespace nn

#endif // MODEL
