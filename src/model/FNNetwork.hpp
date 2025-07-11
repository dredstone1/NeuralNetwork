#ifndef FNNNETWORK
#define FNNNETWORK

#include "../visualizer/FnnVisualizer.hpp"
#include "DenseLayer.hpp"
#include "INetwork.hpp"

namespace nn::model {
class FNNetwork : public INetwork {
  private:
	const FNNConfig &config;
	std::vector<std::unique_ptr<DenseLayer>> layers;
	global::ParamMetrix input;

    void calculateInputDelta(const global::ParamMetrix &deltas);

	const std::shared_ptr<visualizer::FnnVisualier> visual;

  public:
	FNNetwork(
	    const FNNConfig &_config,
	    const bool randomInit,
	    const std::shared_ptr<visualizer::FnnVisualier> visual_);
	~FNNetwork() override = default;

	void forward(const global::ParamMetrix &newInput) override;
    void backward(const global::ParamMetrix &outputDeltas) override;
	void updateWeights(const global::ValueType learningRate) override;
	void resetGradient() override;

	global::ValueType getLoss(const global::Prediction &index) const override;

	int outputSize() const override;
	int inputSize() const override;

	const global::ParamMetrix &getOutput() const override;
	const global::ParamMetrix &getNet() const override;
	const global::ParamMetrix &getInput() const override;

	std::shared_ptr<visualizer::IVisualNetwork> getVisual() override { return visual; }

    global::ParamMetrix getParams() const override;
    void setParams(const global::ParamMetrix params) override;
};
} // namespace nn::model

#endif // FNNNETWORK
