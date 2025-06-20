#ifndef INETWORK
#define INETWORK

#include "config.hpp"
#include <Globals.hpp>

namespace nn::model {
class INetwork {
  public:
	virtual ~INetwork() = default;

	virtual void forward(const global::ParamMetrix &input) = 0;
	virtual void backword(const global::ParamMetrix &output, global::ParamMetrix &deltas) = 0;
	virtual void updateWeights(const global::ValueType learningRate) = 0;
	virtual void resetGradient() = 0;

	virtual int outputSize() const = 0;
	virtual int inputSize() const = 0;

	virtual global::ValueType getLost(const global::ParamMetrix &output) const = 0;

	virtual const global::ParamMetrix &getOutput() const = 0;
};
} // namespace nn::model

#endif // INETWORK
