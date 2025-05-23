#include "layer.hpp"

void Layer::add(const LayerParameters &gradients) {
	Parameters->add(gradients);
}

void Layer::reset() {
	for (int i = 0; i < dots.size(); i++) {
		dots.out[i] = dots.net[i] = 0.0;
	}
}

Layer::~Layer() {
	if (destroyParams) {
		delete Parameters;
	}
}

Layer::Layer(int _size, int _prev_size, int random) : destroyParams(1), dots(_size) {
	Parameters = new LayerParameters(_size, _prev_size, random);
}

const LayerParameters Layer::getParms() {
	return *Parameters;
}
