#ifndef MODEL
#define MODEL

#include "../visualizer/VisualizerController.hpp"
#include "layer.hpp"
#include "neuralNetwork.hpp"

namespace nn {
namespace training {
class BackPropagation;
class Trainer;
} // namespace training

namespace model {
class Model {
  private:
	NeuralNetwork network;
	visualizer::VisualManager visual;

	void runModel(const global::ParamMetrix &input, NeuralNetwork &temp_network);

	friend class training::BackPropagation;
	friend class training::Trainer;

  public:
	Model(Config &_config);
	~Model() = default;

	void runModel(const global::ParamMetrix &input);
	void reset();
	void updateWeights(const training::gradient &gradients);

	Layer &getLayer(const int i) { return *network.layers.at(i); }
	const global::ParamMetrix &getOutput() const;
	size_t getOutputSize() const { return network.config.output_size; }
	size_t getInputSize() const { return network.config.input_size; }
	size_t getHiddenLayerCount() const { return network.config.hidden_layer_count(); }
	size_t getLayerCount() const { return network.config.hidden_layer_count() + 1; }
};
} // namespace model
} // namespace nn

#endif // MODEL
