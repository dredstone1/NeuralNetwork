#include "Hidden_Layer.hpp"
#include "Globals.hpp"
#include <cstddef>
#include <random>

namespace nn {
bool Hidden_Layer::should_drop(float dropOutRate) {
	if (dropOutRate <= 0.0f) {
		return false;
	}

	thread_local static std::mt19937 gen(std::random_device{}());
	std::uniform_real_distribution<> dis(0.0, 1.0);

	return dis(gen) < dropOutRate;
}

void Hidden_Layer::forward(const std::vector<Global::ValueType> &metrix, const Global::ValueType dropOutRate) {
	for (size_t i = 0; i < dots.size(); i++) {
		dots.net[i] = 0;
		if (should_drop(dropOutRate)) {
			dots.out[i] = 0;
			continue;
		}

		for (size_t j = 0; j < metrix.size(); j++) {
			dots.net[i] += Parameters.weights[i][j] * metrix[j];
		}

		dots.out[i] = activate_(dots.net[i]);
	}
}
} // namespace nn
