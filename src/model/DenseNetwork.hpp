#ifndef DENSENEWORK
#define DENSENEWORK

#include "DenseLayer.hpp"
#include "INetwork.hpp"
#include <Globals.hpp>
#include <memory>

namespace nn::model {
class DenseNetwork : INetwork {
  private:
	std::vector<std::unique_ptr<DenseLayer>> layers;

  public:
	DenseNetwork() {}
	virtual ~DenseNetwork() = default;

	void forward(const global::ParamMetrix &input) override;
	void backword(const global::ParamMetrix &output, global::ParamMetrix &deltas) override;
    global::ValueType getLost(const global::ParamMetrix &output) override;
	void updateWeights(const global::ValueType learningRate) override;
};
} // namespace nn::model

#endif // DENSENEWORK
