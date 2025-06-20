#ifndef DENSELAYER
#define DENSELAYER

#include "ILayer.hpp"
#include "LayerParameters.hpp"
#include "activations.hpp"

namespace nn::model {
class DenseLayer : public ILayer {
  protected:
	Neurons dots;
	LayerParameters parameters;

  public:
	DenseLayer(const int size, const int prevSize)
	    : dots(size),
	      parameters(size, prevSize) {}
	virtual ~DenseLayer() = default;

	const Neurons &getDots() const { return dots; }
	global::ValueType getWeight(const int i, const int j) const { return parameters.weights[i][j]; }
	const LayerParameters &getParms() { return parameters; }

	size_t getSize() const { return dots.size(); }
	size_t getPrevSize() const { return parameters.getPrevSize(); }

	const global::ParamMetrix &getNet() const { return dots.net; }
	const global::ParamMetrix &getOut() const { return dots.out; }

	void reset() { dots.reset(); }
	void addParams(const LayerParameters &gradients) { parameters.add(gradients); }
	void setParams(const LayerParameters &gradients) { parameters.set(gradients); }
};

class Hidden_Layer : public DenseLayer {
  private:
	Activation activationFunction;

  public:
	Hidden_Layer(const int _size, const int _prev_size, const ActivationType activation)
	    : DenseLayer(_size, _prev_size),
	      activationFunction(activation) {}

	void forward(const global::ParamMetrix &metrix) override;

	global::ValueType activation(const global::ValueType x) const { return activationFunction.activate(x); }
	global::ValueType derivativeActivation(const global::ValueType x) const { return activationFunction.derivativeActivate(x); }
};

class Output_Layer : public DenseLayer {
  public:
	Output_Layer(const int _size, const int _prev_size)
	    : DenseLayer(_size, _prev_size) {}

	void forward(const global::ParamMetrix &metrix) override;
};
} // namespace nn::model

#endif // DENSELAYER
