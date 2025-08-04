#ifndef FNNNETWORK
#define FNNNETWORK

#include "DenseLayer.hpp"
#include "FnnVisualizer.hpp"
#include "tensor.hpp"
#include <cstddef>
#include <memory>
#include <network/INetwork.hpp>

namespace nn::model::fnn {
class FNNetwork : public INetwork {
  private:
	const FNNConfig &config;
	std::vector<std::unique_ptr<DenseLayer>> layers;
	global::Tensor input;

	const std::shared_ptr<visualizer::fnn::FnnVisualier> visual;

	void calculateInputDelta(const global::Tensor &deltas);

	void vUpdate();

	void sendNewVData(const size_t i) const;
	void sendNewVNeurons(const size_t i) const;

  public:
	FNNetwork(
	    const FNNConfig &_config,
	    const bool randomInit,
	    const std::shared_ptr<visualizer::fnn::FnnVisualier> visual_ = nullptr);
	~FNNetwork() override = default;

	void forward(const global::Tensor &newInput) override;
	void backward(const global::Tensor &outputDeltas) override;
	void updateWeights(IOptimizer &optimizer) override;
	void resetGradient() override;

	global::ValueType getLoss(const global::Prediction &index) const override;

	size_t outputSize() const override;
	size_t inputSize() const override;

	const global::Tensor &getOutput() const override;
	const global::Tensor &getInput() const override;

	std::shared_ptr<visualizer::IVisualNetwork> getVisual() override {
		return visual;
	}

	global::Tensor getParams() const override;
	void setParams(const global::Tensor params) override;

	size_t getParamCount() const;

	void setTraining(const bool state) override;
};
} // namespace nn::model::fnn

#endif // FNNNETWORK
