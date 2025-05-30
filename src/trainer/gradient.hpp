#ifndef GRADIENT
#define GRADIENT

#include "../model/LayerParameters.hpp"
#include "../model/config.hpp"
#include <vector>

struct gradient {
	std::vector<LayerParameters> gradients;
	void add(const gradient &new_gradient);
	void multiply(const double value);
	void reset();
	~gradient() = default;
	const NetworkConfig &config;
	gradient(const NetworkConfig &config);
	gradient(const gradient &other);
};

#endif // GRADIENT
