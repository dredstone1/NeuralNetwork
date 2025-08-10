#ifndef DENSELAYER
#define DENSELAYER

#include "../src/model/optimizers.hpp"
#include <Globals.hpp>
#include <vector>

namespace nn::model::fnn {
constexpr global::ValueType MIN_LOSS_VALUE = 1e-10;

struct LayerParams {
	global::Tensor weights;
	global::Tensor biases;

	size_t size_;
	size_t prevSize_;

	LayerParams(size_t out_dim, size_t in_dim)
	    : weights({out_dim, in_dim}), biases({out_dim}),
	      size_(out_dim), prevSize_(in_dim) {}

	size_t size() const { return size_; }
	size_t prevSize() const { return prevSize_; }

	size_t paramSize() const { return biases.numElements() + weights.numElements(); }
};

class DenseLayer {
  protected:
	global::Tensor net;
	global::Tensor out;

	LayerParams parameters;
	LayerParams gradients;

	global::Tensor deltaL;

	Activation activationFunction;

	bool isTraining{false};

	void fillParamRandom();

  public:
	DenseLayer(
	    const size_t size,
	    const size_t prevSize,
	    const ActivationType activation,
	    const bool randomInit = false);
	virtual ~DenseLayer() = default;

	virtual void forward(const global::Tensor &metrix) = 0;
	void updateWeight(IOptimizer &optimizer);
	virtual void backward(
	    global::Tensor **deltas,
	    const global::Tensor &prevLayer,
	    const LayerParams *nextLayer = nullptr) = 0;
	virtual global::ValueType getLoss(const global::Prediction &) { return 0; };

	const LayerParams &getParms() { return parameters; }
	const LayerParams &getGrad() { return gradients; }

	size_t size() const { return net.numElements(); }
	size_t prevSize() const { return parameters.prevSize(); }

	const global::Tensor &getNet() const { return net; }
	const global::Tensor &getOut() const { return out; }

	void resetDots();
	void resetGradient();

	size_t getParamCount() const;

	const std::vector<global::ValueType> getData() const;
	void setData(const global::Tensor &newParam, const size_t offset);

	void setTraining(const bool state) { isTraining = state; }
};

class Hidden_Layer : public DenseLayer {
  private:
	const DenseLayerConfig &config;
	void calculateDelta(
	    const global::Tensor &output,
	    const LayerParams &nextLayer);

	global::Tensor dropoutMask;

	void CreateDropoutMask();

  public:
	Hidden_Layer(
	    const DenseLayerConfig &_config,
	    const int _prev_size,
	    const bool randomInit = false)
	    : DenseLayer(_config.size, _prev_size, _config.activationType, randomInit),
	      config(_config),
	      dropoutMask({_config.size}) {}
	~Hidden_Layer() override = default;

	void forward(const global::Tensor &metrix) override;
	void backward(
	    global::Tensor **deltas,
	    const global::Tensor &prevLayer,
	    const LayerParams *nextLayer) override;
};

class Output_Layer : public DenseLayer {
  private:
	const FNNConfig &config;

	void getDelta(const global::Tensor &output);
	static global::ValueType getCrossEntropyLoss(
	    const global::Tensor &prediction,
	    const size_t target);

  public:
	Output_Layer(
	    const FNNConfig &_config,
	    const int _prev_size,
	    const bool randomInit = false)
	    : DenseLayer(
	          _config.getOutputSize(),
	          _prev_size,
	          _config.outputActivation,
	          randomInit),
	      config(_config) {}
	~Output_Layer() override = default;

	void forward(const global::Tensor &metrix) override;
	void backward(
	    global::Tensor **deltas,
	    const global::Tensor &prevLayer,
	    const LayerParams *) override;

	global::ValueType getLoss(const global::Prediction &index) override;
};
} // namespace nn::model::fnn

#endif // DENSELAYER
