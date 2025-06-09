#ifndef NEURON
#define NEURON

#include <Globals.hpp>

namespace nn::model {
struct Neurons {
	global::ParamMetrix out;
	global::ParamMetrix net;

	Neurons(const int size);
	Neurons(const global::ParamMetrix &net_, const global::ParamMetrix &out_) : out(out_), net(net_) {}
	~Neurons() = default;
	Neurons(const Neurons &other)
	    : out(other.out),
	      net(other.net) {}
	size_t size() const { return out.size(); }
	void reset();
};
} // namespace nn::model
#endif // NEURON
