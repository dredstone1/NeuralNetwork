#ifndef NEURON
#define NEURON

#include "Globals.hpp"
#include <vector>

struct neurons {
	std::vector<Global::ValueType> out;
	std::vector<Global::ValueType> net;
	neurons(const int size);
	~neurons() = default;
	neurons(const neurons &other)
	    : out(other.out),
	      net(other.net) {}
	int size() const { return out.size(); }
	void reset();
};

#endif // NEURON
