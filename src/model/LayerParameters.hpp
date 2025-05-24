#ifndef LAYER_PARAMETERS_HPP
#define LAYER_PARAMETERS_HPP

#include <vector>

using namespace std;

struct LayerParameters {
	vector<vector<double>> weights;
	LayerParameters(const int size, const int prev_size, const bool random = false);
	LayerParameters(LayerParameters const &other);
	~LayerParameters() = default;
	int getSize() const { return weights.size(); }
	int getPrevSize() const { return weights[0].size(); }
	void add(const LayerParameters &new_gradient);
	void multiply(const double value);
	void reset();
};

#endif // LAYER_PARAMETERS_HPP
