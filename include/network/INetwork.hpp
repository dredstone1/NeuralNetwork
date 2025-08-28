#ifndef INETWORK
#define INETWORK

#include "../src/model/optimizers.hpp"
#include "IvisualNetwork.hpp"
#include "tensor.hpp"

namespace nn::model {
class INetwork {
  public:
	virtual ~INetwork() = default;

	virtual void forward(const global::Tensor &input) = 0;
	virtual void backward(global::Tensor **outputDeltas, const global::ValueType weight) = 0;
	virtual void updateWeights(IOptimizer &optimizer) = 0;
	virtual void resetGradient() = 0;

	virtual size_t outputSize() const = 0;

	virtual global::ValueType getLoss(const global::Prediction &index) const = 0;
	virtual const global::Tensor &getOutput() const = 0;
	virtual global::Tensor *getInput() = 0;
	virtual std::shared_ptr<visualizer::IVisualNetwork> getVisual() = 0;

	virtual std::vector<global::ValueType> getParams() const = 0;
	virtual void setParams(const global::Tensor &params) = 0;

	virtual void setTraining(const bool state) = 0;

	virtual size_t getParamCount() const = 0;
};
} // namespace nn::model

#endif // INETWORK
