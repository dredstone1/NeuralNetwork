#ifndef FNNNETWORK
#define FNNNETWORK

#include "DenseLayer.hpp"
#include "INetwork.hpp"

namespace nn::model {
class FNNetwork : public INetwork {
  private:
	std::vector<std::unique_ptr<DenseLayer>> layers;
	const FNNConfig &config;

  public:
	FNNetwork(const FNNConfig &_config, const bool randomInit = false);
	~FNNetwork() override = default;

	void forward(const global::ParamMetrix &input) override;
	void backword(const global::ParamMetrix &output, global::ParamMetrix &deltas) override;
	void updateWeights(const global::ValueType learningRate) override;
	void resetGradient() override;

	global::ValueType getLost(const int index) const override;

	int outputSize() const override;
	int inputSize() const override;

	const global::ParamMetrix &getOutput() const override;
};
} // namespace nn::model

#endif // FNNNETWORK
