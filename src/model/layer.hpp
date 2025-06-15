#ifndef LAYER
#define LAYER

#include "LayerParameters.hpp"
#include "activations.hpp"
#include "neuron.hpp"

namespace nn::model {
enum class LayerType {
	HIDDEN,
	OUTPUT,
	NONE,
};

class Layer {
  protected:
	Neurons dots;
	LayerParameters parameters;

	friend class Model;

  public:
	Layer(const int size, const int prevSize, const global::ValueType initValue)
	    : dots(size),
	      parameters(size, prevSize, initValue) {}

	virtual LayerType getType() const { return LayerType::NONE; }
	virtual void forward(const global::ParamMetrix &metrix);
	virtual ~Layer() = default;

	const Neurons &getDots() const { return dots; }
	global::ValueType getWeight(const int i, const int j) const { return parameters.weights[i][j]; }
	const LayerParameters getParms() { return parameters; }

	size_t getSize() const { return dots.size(); }
	size_t getPrevSize() const { return parameters.getPrevSize(); }

	const global::ParamMetrix &getNet() const { return dots.net; }
	const global::ParamMetrix &getOut() const { return dots.out; }

	void setWeight(const int i, const int j, const global::ValueType weight) { parameters.weights[i][j] = weight; }
	void reset() { dots.reset(); }
	void addParams(const LayerParameters &gradients) { parameters.add(gradients); }
	void setParams(const LayerParameters &gradients) { parameters.set(gradients); }
};

class Hidden_Layer : public Layer {
  private:
	Activation activationFunction;

  public:
	Hidden_Layer(const int _size, const int _prev_size, const ActivationType activation, const global::ValueType init_value)
	    : Layer(_size, _prev_size, init_value),
	      activationFunction(activation) {}

	void forward(const global::ParamMetrix &metrix) override;
	LayerType getType() const override { return LayerType::HIDDEN; }

	global::ValueType activation(const global::ValueType x) const { return activationFunction.activate(x); }
	global::ValueType derivativeActivation(const global::ValueType x) const { return activationFunction.derivativeActivate(x); }
};

class Output_Layer : public Layer {
  public:
	Output_Layer(const int _size, const int _prev_size, const global::ValueType init_value)
	    : Layer(_size, _prev_size, init_value) {}
	void forward(const global::ParamMetrix &metrix) override;
	LayerType getType() const override { return LayerType::OUTPUT; }
};
} // namespace nn::model

#endif // LAYER
