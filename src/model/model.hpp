#ifndef MODEL
#define MODEL

#include "../visualizer/VisualizerController.hpp"
#include "Layers/layer.hpp"
#include "config.hpp"
#include "neuralNetwork.hpp"

namespace nn::training {
class BackPropagation;
class Trainer;
} // namespace nn::training

namespace nn::model {
class Model {
  private:
	NeuralNetwork network;
	visualizer::visualizerController visual;
	void runModel(const global::ParamMetrix &input, NeuralNetwork &temp_network);
	friend class training::BackPropagation;
	friend class training::Trainer;

  public:
	Model(Config &_config, bool use_visual);
	~Model() = default;
	void runModel(const global::ParamMetrix &input);
	const global::ParamMetrix &getOutput() const;
	void reset();
	Layer &getLayer(const int i) { return *network.layers.at(i); }
	void updateWeights(const training::gradient &gradients);
	size_t getOutputSize() const { return network.config.output_size; }
	size_t getInputSize() const { return network.config.input_size; }
	size_t getHiddenLayerCount() const { return network.config.hidden_layer_count(); }
	size_t getLayerCount() const { return network.config.hidden_layer_count() + 1; }
	const bool useVisual;
};
} // namespace nn::model

#endif // MODEL
