#ifndef INETWORK
#define INETWORK

#include <Globals.hpp>

namespace nn::model {
class INetwork {
  public:
	INetwork() {}
	virtual ~INetwork() = default;

	virtual void forward(const global::ParamMetrix &input);
	virtual void backword(const global::ParamMetrix &output, global::ParamMetrix &deltas);
	virtual void updateWeights(const global::ValueType learningRate);
	virtual void resetGradient();

    virtual int outputSize();
    virtual int inputSize();

    virtual global::ValueType getLost(const global::ParamMetrix &output);

    virtual global::ParamMetrix& getOutput();
};
} // namespace nn::model

#endif // INETWORK
