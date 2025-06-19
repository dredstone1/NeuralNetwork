#ifndef DENSENEWORK
#define DENSENEWORK

#include "ILayer.hpp"
#include "INetwork.hpp"
#include <Globals.hpp>
#include <memory>

namespace nn::model {
class DenseNetwork : INetwork {
  private:
	std::vector<std::unique_ptr<ILayer>> layers;

  public:
	DenseNetwork() {}
	virtual ~DenseNetwork() = default;

	virtual void forward(const global::ParamMetrix &input, global::ParamMetrix &output);
	virtual void backword(const global::ParamMetrix &output, global::ParamMetrix &input);
	virtual void updateWeights(const global::ValueType learningRate);
};
} // namespace nn::model

#endif // DENSENEWORK
