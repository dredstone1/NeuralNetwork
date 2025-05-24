#ifndef GRADIENT_HPP
#define GRADIENT_HPP

#include "../model/LayerParameters.hpp"
#include "../model/config.hpp"
#include <vector>

using namespace std;

typedef struct gradient {
	vector<LayerParameters> gradients;
	void add(const gradient &new_gradient);
	void multiply(const double value);
	void reset();
	~gradient() = default;
    NetworkConfig &config;
	gradient(NetworkConfig &config);
	gradient(const gradient &other);
} gradient;

#endif // GRADIENT_HPP
