#ifndef FNNNETWORK
#define FNNNETWORK

#include "DenseLayer.hpp"
#include "INetwork.hpp"

namespace nn::model {
class FNNetwork : public INetwork {
  private:
	const FNNConfig &config;
	std::vector<std::unique_ptr<DenseLayer>> layers;
	global::ParamMetrix input;

  public:
	FNNetwork(const FNNConfig &_config, const bool randomInit = false);
	~FNNetwork() override = default;

	void forward(const global::ParamMetrix &newInput) override;
	void backward(global::ParamMetrix &deltas) override;
	void updateWeights(const global::ValueType learningRate) override;
	void resetGradient() override;

	global::ValueType getLoss(const int index) const override;

	int outputSize() const override;
	int inputSize() const override;

	const global::ParamMetrix &getOutput() const override;
};
} // namespace nn::model

#endif // FNNNETWORK
