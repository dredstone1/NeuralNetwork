#ifndef NEURON
#define NEURON

#include <vector>

struct neurons {
    std::vector<double> out;
    std::vector<double> net;
	neurons(const int size);
	~neurons() = default;
	int size() const { return out.size(); }
	void reset();
};

#endif // NEURON
