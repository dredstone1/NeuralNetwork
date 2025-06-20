#ifndef INETWORK
#define INETWORK

#include <Globals.hpp>

namespace nn::model {
class INetwork {
  public:
	INetwork() {}
	virtual ~INetwork() = default;

	virtual void forward(const global::ParamMetrix &input);
	virtual void backword(const global::ParamMetrix &output, global::ParamMetrix &input);
	virtual void updateWeights(const global::ValueType learningRate);

    virtual int outputSize();
    virtual int inputSize();
    
    virtual global::ParamMetrix& getOutput();
};
} // namespace nn::model

#endif // INETWORK
