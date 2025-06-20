#ifndef DENSENEWORK
#define DENSENEWORK

#include "DenseLayer.hpp"
#include "INetwork.hpp"
#include <Globals.hpp>
#include <memory>
#include <vector>

namespace nn::model {
struct DenseLayerConfig {
	int size;
	global::ValueType weights_init_value = -1;
	ActivationType AT = ActivationType::LeakyRelu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    DenseLayerConfig,
    size,
    AT,
    weights_init_value);

class DenseNetwork : INetwork {
  private:
	std::vector<std::unique_ptr<DenseLayer>> layers;

  public:
	virtual ~DenseNetwork() = default;

	void forward(const global::ParamMetrix &input) override;
	void backword(const global::ParamMetrix &output, global::ParamMetrix &deltas) override;
	void updateWeights(const global::ValueType learningRate) override;
    void resetGradient() override;

	global::ValueType getLost(const global::ParamMetrix &output) const override;

    int outputSize() const override;
    int inputSize() const override;

	void fromJson(const nlohmann::json &j) override;
};
} // namespace nn::model

#endif // DENSENEWORK
