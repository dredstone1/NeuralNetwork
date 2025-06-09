#ifndef LAYER
#define LAYER

#include "../LayerParameters.hpp"
#include "../neuron.hpp"

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
	Layer(const Layer &other)
	    : dots(other.dots.size()),
	      parameters(other.parameters) {}
	Layer(const int size, const int prevSize, const global::ValueType initValue)
	    : dots(size),
	      parameters(size, prevSize, initValue) {}
	virtual LayerType getType() const { return LayerType::NONE; }
	virtual void forward(const global::ParamMetrix &metrix);
	const Neurons &getDots() const { return dots; }
	global::ValueType getWeight(const int i, const int j) const { return parameters.weights[i][j]; }
	void setWeight(const int i, const int j, const global::ValueType weight) { parameters.weights[i][j] = weight; }
	const global::ParamMetrix &getNet() const { return dots.net; }
	const global::ParamMetrix &getOut() const { return dots.out; }
	void addParams(const LayerParameters &gradients) { parameters.add(gradients); }
	size_t getSize() const { return dots.size(); }
	size_t getPrevSize() const { return parameters.getPrevSize(); }
	void reset() { dots.reset(); }
	const LayerParameters getParms() { return parameters; }
	virtual ~Layer() = default;
};
} // namespace nn::model
#endif // LAYER
