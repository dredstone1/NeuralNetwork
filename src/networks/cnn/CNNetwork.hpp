#ifndef CNNNETWORK
#define CNNNETWORK

#include "CnnVisualizer.hpp"
#include <memory>
#include <network/INetwork.hpp>

namespace nn::model::cnn {
class CNNetwork : public INetwork {
  private:
	const CNNConfig &config;
	global::ParamMetrix input;

	void calculateInputDelta(const global::ParamMetrix &deltas);

	const std::shared_ptr<visualizer::cnn::CnnVisualier> visual;

  public:
	CNNetwork(
	    const CNNConfig &_config,
	    const bool randomInit,
	    const std::shared_ptr<visualizer::cnn::CnnVisualier> visual_ = std::shared_ptr<visualizer::cnn::CnnVisualier>());
	~CNNetwork() override = default;

	void forward(const global::ParamMetrix &newInput) override;
	void backward(const global::ParamMetrix &outputDeltas) override;
	void updateWeights(IOptimizer &optimizer) override;
	void resetGradient() override;

	global::ValueType getLoss(const global::Prediction &index) const override;

	int outputSize() const override;
	int inputSize() const override;

	const global::ParamMetrix &getOutput() const override;
	const global::ParamMetrix &getInput() const override;

	std::shared_ptr<visualizer::IVisualNetwork> getVisual() override { return visual; }

	global::ParamMetrix getParams() const override;
	void setParams(const global::ParamMetrix params) override;

	void setTraining(const bool state) override;
};
} // namespace nn::model::cnn

#endif // CNNNETWORK
