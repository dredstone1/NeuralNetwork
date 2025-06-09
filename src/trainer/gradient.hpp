#ifndef GRADIENT
#define GRADIENT

#include "../model/LayerParameters.hpp"
#include "../model/config.hpp"

namespace nn::training {
struct gradient {
	std::vector<model::LayerParameters> gradients;
	const model::NetworkConfig &config;

	void add(const gradient &new_gradient);
	void multiply(const global::ValueType value);
	void reset();
	~gradient() = default;
	gradient(const model::NetworkConfig &config);
	gradient(const gradient &other);
};
} // namespace nn::training

#endif // GRADIENT
