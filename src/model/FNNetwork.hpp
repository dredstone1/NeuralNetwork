#ifndef FNNNETWORK
#define FNNNETWORK

#include "DenseLayer.hpp"
#include "INetwork.hpp"
#include "config.hpp"
#include <Globals.hpp>
#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <vector>

namespace nn::model {
class DenseNetwork : public INetwork {
  private:
	std::vector<std::unique_ptr<DenseLayer>> layers;
	const FNNConfig &config;

  public:
	DenseNetwork(const FNNConfig &_config) : config(_config) {}
	virtual ~DenseNetwork() = default;

	void forward(const global::ParamMetrix &input) override;
	void backword(const global::ParamMetrix &output, global::ParamMetrix &deltas) override;
	void updateWeights(const global::ValueType learningRate) override;
	void resetGradient() override;

	global::ValueType getLost(const global::ParamMetrix &output) const override;

	int outputSize() const override;
	int inputSize() const override;

	const global::ParamMetrix &getOutput() const override;
};
} // namespace nn::model

#endif // FNNNETWORK
