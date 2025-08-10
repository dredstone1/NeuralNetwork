#ifndef CNNNETWORK
#define CNNNETWORK

#include "CnnVisualizer.hpp"
#include "tensor.hpp"
#include <memory>
#include <network/INetwork.hpp>
#include <vector>

namespace nn::model::cnn {
class CNNetwork : public INetwork {
  private:
	const CNNConfig &config;
	global::Tensor input;

	void calculateInputDelta(const global::Tensor &deltas);

	const std::shared_ptr<visualizer::cnn::CnnVisualier> visual;

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
	size_t inputSize() const override;

	const global::Tensor &getOutput() const override;
	const global::Tensor &getInput() const override;

	std::shared_ptr<visualizer::IVisualNetwork> getVisual() override { return visual; }

    std::vector<global::ValueType> getParams() const override;
	void setParams(const global::Tensor params) override;

	void setTraining(const bool state) override;
};
} // namespace nn::model::cnn

#endif // CNNNETWORK
