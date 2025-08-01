#ifndef DENSELAYER
#define DENSELAYER

#include "../../model/config.hpp"
#include "../src/model/optimizers.hpp"
#include "Globals.hpp"
#include "LayerParameters.hpp"

namespace nn::model::fnn {
constexpr global::ValueType MIN_LOSS_VALUE = 1e-10;

struct Neurons {
	global::ParamMetrix out;
	global::ParamMetrix net;

	Neurons(const int size);
	Neurons(const global::ParamMetrix &net_, const global::ParamMetrix &out_)
	    : out(out_),
	      net(net_) {}
	~Neurons() = default;

	Neurons(const Neurons &other) : out(other.out), net(other.net) {}
	size_t size() const { return out.size(); }
	void reset();
};

class DenseLayer {
  protected:
	Neurons dots;
	LayerParameters parameters;
	LayerParameters gradients;

	Activation activationFunction;

	bool isTraining{false};

  public:
	DenseLayer(
	    const int size,
	    const int prevSize,
	    const ActivationType activation,
	    const bool randomInit = false);
	virtual ~DenseLayer() = default;

	virtual void forward(const global::ParamMetrix &metrix) = 0;
	void updateWeight(IOptimizer &optimizer);
	virtual void backward(
	    global::ParamMetrix &deltas,
	    const global::ParamMetrix &prevLayer,
	    const LayerParameters *nextLayer = nullptr) = 0;
	virtual global::ValueType getLoss(const global::Prediction &) { return 0; };

	const Neurons &getDots() const { return dots; }
	const LayerParameters &getParms() { return parameters; }
	const LayerParameters &getGrad() { return gradients; }

	size_t getSize() const { return dots.size(); }
	size_t getPrevSize() const { return parameters.getPrevSize(); }

	const global::ParamMetrix &getNet() const { return dots.net; }
	const global::ParamMetrix &getOut() const { return dots.out; }

	void resetDots() { dots.reset(); }
	void resetGradient() { gradients.reset(); }
	void addParams(const LayerParameters &gradients) { parameters.add(gradients); }
	void setParams(const LayerParameters &gradients) { parameters.set(gradients); }

	const global::ParamMetrix getData() const;
	void setData(const global::ParamMetrix newParam);

	void setTraining(const bool state) { isTraining = state; }
};

class Hidden_Layer : public DenseLayer {
  private:
	const DenseLayerConfig &config;
	global::ParamMetrix getDelta(
	    const global::ParamMetrix &output,
	    const LayerParameters &nextLayer);

	std::vector<int> dropoutMask;

	void CreateDropoutMask();

  public:
	Hidden_Layer(
	    const DenseLayerConfig &_config,
	    const int _prev_size,
	    const bool randomInit = false)
	    : DenseLayer(_config.size, _prev_size, _config.activationType, randomInit),
	      config(_config) {}
	~Hidden_Layer() override = default;

	void forward(const global::ParamMetrix &metrix) override;
	void backward(
	    global::ParamMetrix &deltas,
	    const global::ParamMetrix &prevLayer,
	    const LayerParameters *nextLayer) override;
};

class Output_Layer : public DenseLayer {
  private:
	const FNNConfig &config;

	global::ParamMetrix getDelta(const global::ParamMetrix &output);
	static global::ValueType getCrossEntropyLoss(
	    const global::ParamMetrix &prediction,
	    const int target);

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

	void forward(const global::ParamMetrix &metrix) override;
	void backward(
	    global::ParamMetrix &deltas,
	    const global::ParamMetrix &prevLayer,
	    const LayerParameters *) override;

	global::ValueType getLoss(const global::Prediction &index) override;
};

} // namespace nn::model::fnn

#endif // DENSELAYER
