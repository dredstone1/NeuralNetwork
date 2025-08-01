#ifndef FNNNETWORK
#define FNNNETWORK

#include "DenseLayer.hpp"
#include "FnnVisualizer.hpp"
#include <cstddef>
#include <memory>
#include <network/INetwork.hpp>

namespace nn::model::fnn {
class FNNetwork : public INetwork {
  private:
	const FNNConfig &config;
	std::vector<std::unique_ptr<DenseLayer>> layers;
	global::ParamMetrix input;

	const std::shared_ptr<visualizer::fnn::FnnVisualier> visual;

	void calculateInputDelta(const global::ParamMetrix &deltas);

	void Vinit(const size_t i);

  public:
	FNNetwork(
	    const FNNConfig &_config,
	    const bool randomInit,
	    const std::shared_ptr<visualizer::fnn::FnnVisualier> visual_ = nullptr);
	~FNNetwork() override = default;

	void forward(const global::ParamMetrix &newInput) override;
	void backward(const global::ParamMetrix &outputDeltas) override;
	void updateWeights(IOptimizer &optimizer) override;
	void resetGradient() override;

	global::ValueType getLoss(const global::Prediction &index) const override;

	int outputSize() const override;
	int inputSize() const override;

	const global::ParamMetrix &getOutput() const override;
	const global::ParamMetrix &getInput() const override;

	std::shared_ptr<visualizer::IVisualNetwork> getVisual() override {
		return visual;
	}

	global::ParamMetrix getParams() const override;
	void setParams(const global::ParamMetrix params) override;

	void setTraining(const bool state) override;
};
} // namespace nn::model::fnn

#endif // FNNNETWORK
