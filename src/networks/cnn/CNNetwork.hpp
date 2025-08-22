#ifndef CNNNETWORK
#define CNNNETWORK

#include "CnnVisualizer.hpp"
#include <network/INetwork.hpp>

namespace nn::model::cnn {

struct Size {
	size_t w;
	size_t h;
};

class CNNetwork : public INetwork {
  private:
	const CNNConfig &config;
	global::Tensor input;

	global::Tensor filtersW;
	global::Tensor filtersWGradient;

	global::Tensor filtersB;
	global::Tensor filtersBGradient;

	global::Tensor activationMapN;
	global::Tensor activationMapO;

	global::Tensor output;

	global::Tensor inputDelta;
	global::Tensor activationDelta;

	Activation activationFunction;

	void calculateInputDelta(const global::Tensor &deltas);
	void calculateFilterGradients();
	void calculateBiasGradients();
	void initializeParameters();

	const std::shared_ptr<visualizer::cnn::CnnVisualier> visual;

	std::vector<size_t> makeActivationMapShape();

	std::vector<global::ValueType> randomFilters() const;

	void conv2d_cpu();

	Size getFeatureMapSize();

  public:
	CNNetwork(
	    const CNNConfig &_config,
	    const bool randomInit,
	    const std::shared_ptr<visualizer::cnn::CnnVisualier> visual_ = std::shared_ptr<visualizer::cnn::CnnVisualier>());
	~CNNetwork() override = default;

	void forward(const global::Tensor &newInput) override;
	void backward(global::Tensor **outputDeltas) override;
	void updateWeights(IOptimizer &optimizer) override;
	void resetGradient() override;

	global::ValueType getLoss(const global::Prediction &index) const override;

	size_t outputSize() const override;

	const global::Tensor &getOutput() const override;
	const global::Tensor &getInput() const override;
	const global::Tensor &getInputDelta() const { return inputDelta; }

	std::shared_ptr<visualizer::IVisualNetwork> getVisual() override { return visual; }

	std::vector<global::ValueType> getParams() const override;
	void setParams(const global::Tensor &params) override;

	size_t getParamCount() const override;

	void setTraining(const bool) override {}
};

} // namespace nn::model::cnn

#endif // CNNNETWORK
