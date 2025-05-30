#ifndef LAYER
#define LAYER

#include "../LayerParameters.hpp"
#include "../neuron.hpp"

enum LayerType {
	HIDDEN,
	OUTPUT,
	NONE
};

class Layer {
  protected:
	neurons dots;
	LayerParameters Parameters;

  public:
	Layer(const Layer &other)
	    : dots(other.dots.size()),
	      Parameters(other.Parameters) {}
	Layer(const int _size, const int _prev_size, const double init_value)
	    : dots(_size),
	      Parameters(_size, _prev_size, init_value) {}
	virtual LayerType getType() const { return NONE; }
	virtual void forward(const std::vector<double> &metrix);
	const neurons &getDots() const { return dots; }
	double getWeight(const int i, const int j) const { return Parameters.weights[i][j]; }
	void setWeight(const int i, const int j, const double weight) { Parameters.weights[i][j] = weight; }
	const std::vector<double> &getNet() const { return dots.net; }
	const std::vector<double> &getOut() const { return dots.out; }
	void add(const LayerParameters &gradients);
	int getSize() const { return dots.size(); }
	int getPrevSize() const { return Parameters.getPrevSize(); }
	void reset();
	const LayerParameters getParms();
	virtual ~Layer() = default;
};

#endif // LAYER
