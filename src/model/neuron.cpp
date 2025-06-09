#include "neuron.hpp"

namespace nn::model {
void Neurons::reset() {
	for (size_t i = 0; i < out.size(); i++) {
		out[i] = 0.0;
		net[i] = 0.0;
	}
}

Neurons::Neurons(const int size) {
	out.resize(size, 0.0);
	net.resize(size, 0.0);
}
} // namespace nn::model
