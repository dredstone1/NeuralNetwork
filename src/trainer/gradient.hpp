#ifndef GRADIENT_HPP
#define GRADIENT_HPP

#include "../model/LayerParameters.hpp"
#include <vector>

using namespace std;

typedef struct gradient {
	vector<LayerParameters> gradients;
	void add(const gradient &new_gradient);
	void multiply(const double value);
	void reset();
	~gradient() = default;
	gradient(const int input_size, const int output_size, const int hidden_layers_size, const int hidden_layers_count);
	gradient(const gradient &other);
} gradient;

#endif // GRADIENT_HPP
