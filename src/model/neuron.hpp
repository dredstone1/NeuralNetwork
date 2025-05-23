#ifndef NEURON
#define NEURON

#include <vector>

using namespace std;

struct neurons {
	vector<double> out;
	vector<double> net;
	neurons(const int size);
	~neurons() = default;
	int size() const { return out.size(); }
	void reset();
};

#endif // NEURON
