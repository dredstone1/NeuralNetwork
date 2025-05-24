#ifndef MODEL_HPP
#define MODEL_HPP

#include "../trainer/gradient.hpp"
#include "../visualizer/VisualizerController.hpp"
#include "Layers/layer.hpp"
#include "config.hpp"
#include "neuralNetwork.hpp"

using namespace Visualizer;

class model {
  private:
	neural_network network;
	visualizerController visual;
	void run_model(const vector<double> &input, neural_network &temp_network);
	friend class BackPropagation;

  public:
	model(Config &_config, bool use_visual);
	~model() = default;
	void run_model(const vector<double> &input);
	const vector<double> &getOutput() const;
	void reset();
	Layer &getLayer(const int i) { return *(network.layers[i]); }
	void updateWeights(const gradient &gradients);
	int getOutputSize() const { return network.config.output_size; }
	int getInputSize() const { return network.config.input_size; }
	int getHiddenLayerCount() const { return network.config.hidden_layer_count(); }
	int getLayerCount() const { return network.config.hidden_layer_count() + 1; }
	const bool useVisual;
};

#endif // MODEL_HPP
