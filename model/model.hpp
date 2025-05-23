#ifndef MODEL_HPP
#define MODEL_HPP

#include "../trainer/gradient.hpp"
#include "../visualizer/VisualizerController.hpp"
#include "Layers/layer.hpp"
#include "neuralNetwork.hpp"

using namespace Visualizer;

class model {
  private:
	neural_network network;
	visualizerController visual;
	void run_model(const vector<double> &input, neural_network &temp_network);
	friend class BackPropagation;

  public:
	model(const int input_size, const int output_size, const int hidden_layers_size, const int hidden_layers_count);
	~model() = default;
	void run_model(const vector<double> &input);
	const vector<double> &getOutput() const;
	void reset();
	Layer &getLayer(const int i) { return *(network.layers[i]); }
	void updateWeights(const gradient &gradients);
	int getOutputSize() const { return network.output_size; }
	int getInputSize() const { return network.input_size; }
	int getHiddenLayerSize() const { return network.hidden_layers_size; }
	int getHiddenLayerCount() const { return network.hidden_layers_count; }
	int getLayerCount() const { return network.hidden_layers_count + 1; }
};

#endif // MODEL_HPP
