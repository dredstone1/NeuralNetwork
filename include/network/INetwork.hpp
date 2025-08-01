#ifndef INETWORK
#define INETWORK

#include "../src/model/optimizers.hpp"
#include "IvisualNetwork.hpp"

namespace nn::model {
class INetwork {
  public:
	virtual ~INetwork() = default;

	virtual void forward(const global::ParamMetrix &input) = 0;
	virtual void backward(const global::ParamMetrix &outputDeltas) = 0;
	virtual void updateWeights(IOptimizer &optimizer) = 0;
	virtual void resetGradient() = 0;

	virtual int outputSize() const = 0;
	virtual int inputSize() const = 0;

	virtual global::ValueType getLoss(const global::Prediction &index) const = 0;
	virtual const global::ParamMetrix &getOutput() const = 0;
	virtual const global::ParamMetrix &getInput() const = 0;
	virtual std::shared_ptr<visualizer::IVisualNetwork> getVisual() = 0;

	virtual global::ParamMetrix getParams() const = 0;
	virtual void setParams(const global::ParamMetrix params) = 0;

	virtual void setTraining(const bool state) = 0;
};
} // namespace nn::model

#endif // INETWORK
