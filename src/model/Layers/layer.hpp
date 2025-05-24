#ifndef LAYER_HPP
#define LAYER_HPP

#include "../LayerParameters.hpp"
#include "../neuron.hpp"

enum LayerType {
	HIDDEN,
	OUTPUT,
	NONE
};

class Layer {
  private:
	const int destroyParams;

  protected:
	neurons dots;
	LayerParameters *Parameters;

  public:
	Layer(Layer const &other) : destroyParams(1), dots(other.dots.size()), Parameters(other.Parameters) {}
	Layer(int _size, int _prev_size, int random = false);
	virtual LayerType getType() const { return NONE; }
	virtual void forward(const vector<double> &metrix);
	const neurons &getDots() const { return dots; }
	double getWeight(int i, int j) const { return Parameters->weights[i][j]; }
	void setWeight(int i, int j, double weight) { Parameters->weights[i][j] = weight; }
	double getBias(int i) { return Parameters->bias[i]; }
	void setBias(int i, double bias) { Parameters->bias[i] = bias; }
	const vector<double> &getNet() const { return dots.net; }
	const vector<double> &getOut() const { return dots.out; }
	void add(LayerParameters const &gradients);
	int getSize() const { return Parameters->getSize(); }
	int getPrevSize() const { return Parameters->getPrevSize(); }
	void reset();
	const LayerParameters getParms();
	virtual ~Layer();
};

#endif // LAYER_HPP
