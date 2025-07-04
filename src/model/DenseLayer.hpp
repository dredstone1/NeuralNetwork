#ifndef DENSELAYER
#define DENSELAYER

#include "Globals.hpp"
#include "LayerParameters.hpp"
#include "activations.hpp"
#include "config.hpp"

namespace nn::model {
constexpr global::ValueType MIN_LOSS_VALUE = 1e-10;

class DenseLayer {
  private:
	void randomize();

  protected:
	Neurons dots;
	LayerParameters parameters;
	LayerParameters gradients;

  public:
	DenseLayer(const int size, const int prevSize, const bool randomInit = false)
	    : dots(size),
	      parameters(size, prevSize),
	      gradients(size, prevSize) {
		if (randomInit)
			randomize();
	}
	virtual ~DenseLayer() = default;

	virtual void forward(const global::ParamMetrix &metrix) = 0;
	void updateWeight(const global::ValueType learningRate);
	virtual void backward(
	    global::ParamMetrix &deltas,
	    const global::ParamMetrix &prevLayer,
	    const LayerParameters *nextLayer = nullptr) = 0;
	virtual global::ValueType getLoss(const int index) = 0;

	const Neurons &getDots() const { return dots; }
	global::ValueType getWeight(const int i, const int j) const { return parameters.weights[i][j]; }
	const LayerParameters &getParms() { return parameters; }

	size_t getSize() const { return dots.size(); }
	size_t getPrevSize() const { return parameters.getPrevSize(); }

	const global::ParamMetrix &getNet() const { return dots.net; }
	const global::ParamMetrix &getOut() const { return dots.out; }

	void resetDots() { dots.reset(); }
	void resetGradient() { gradients.reset(); }
	void addParams(const LayerParameters &gradients) { parameters.add(gradients); }
	void setParams(const LayerParameters &gradients) { parameters.set(gradients); }
};

class Hidden_Layer : public DenseLayer {
  private:
	Activation activationFunction;
	global::ParamMetrix getDelta(const global::ParamMetrix &output, const LayerParameters &nextLayer);

  public:
	Hidden_Layer(
	    const int _size,
	    const int _prev_size,
	    const ActivationType activation, const bool randomInit = false)
	    : DenseLayer(_size, _prev_size, randomInit),
	      activationFunction(activation) {}

	Hidden_Layer(const DenseLayerConfig &_config, const int _prev_size, const bool randomInit = false)
	    : DenseLayer(_config.size, _prev_size, randomInit),
	      activationFunction(_config.activationType) {}
	~Hidden_Layer() override = default;

	void forward(const global::ParamMetrix &metrix) override;
	void backward(
	    global::ParamMetrix &deltas,
	    const global::ParamMetrix &prevLayer,
	    const LayerParameters *nextLayer) override;

	global::ValueType getLoss(const int index) override;

	global::ValueType activation(const global::ValueType x) const {
		return activationFunction.activate(x);
	}
	global::ValueType derivativeActivation(const global::ValueType x) const {
		return activationFunction.derivativeActivate(x);
	}
};

class Output_Layer : public DenseLayer {
  private:
	global::ParamMetrix getDelta(const global::ParamMetrix &output);
	static global::ValueType getCrossEntropyLoss(const global::ParamMetrix &prediction, const int target);

  public:
	Output_Layer(const int _size, const int _prev_size, const bool randomInit = false)
	    : DenseLayer(_size, _prev_size, randomInit) {}
	Output_Layer(const FNNConfig &_config, const int _prev_size, const bool randomInit = false)
	    : DenseLayer(_config.getOutputSize(), _prev_size, randomInit) {}
	~Output_Layer() override = default;

	void forward(const global::ParamMetrix &metrix) override;
	void backward(
	    global::ParamMetrix &deltas,
	    const global::ParamMetrix &prevLayer,
	    const LayerParameters *) override;
	global::ValueType getLoss(const int index) override;
};
} // namespace nn::model

#endif // DENSELAYER
