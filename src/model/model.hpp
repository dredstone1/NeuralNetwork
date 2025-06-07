#ifndef MODEL
#define MODEL

#include "../visualizer/VisualizerController.hpp"
#include "Layers/layer.hpp"
#include "config.hpp"
#include "neuralNetwork.hpp"

namespace nn {
class model {
  private:
	neural_network network;
	Visualizer::visualizerController visual;
	void run_model(const std::vector<Global::ValueType> &input, neural_network &temp_network);
	friend class BackPropagation;
	friend class Trainer;

  public:
	model(Config &_config, bool use_visual);
	~model() = default;
	void run_model(const std::vector<Global::ValueType> &input);
	const std::vector<Global::ValueType> &getOutput() const;
	void reset();
	Layer &getLayer(const int i) { return *network.layers.at(i); }
	void updateWeights(const gradient &gradients);
	size_t getOutputSize() const { return network.config.output_size; }
	size_t getInputSize() const { return network.config.input_size; }
	size_t getHiddenLayerCount() const { return network.config.hidden_layer_count(); }
	size_t getLayerCount() const { return network.config.hidden_layer_count() + 1; }
	const bool useVisual;
};
} // namespace nn

#endif // MODEL
