#ifndef LAYER_PARAMETERS
#define LAYER_PARAMETERS

#include <vector>

struct LayerParameters {
	std::vector<std::vector<double>> weights;
	LayerParameters(const int size, const int prev_size, const double init_value);
	LayerParameters(const LayerParameters &other);
	~LayerParameters() = default;
	int getSize() const {
		return weights.size();
	}
	int getPrevSize() const {
		if (!weights.empty() && !weights[0].empty()) {
			return weights[0].size();
		}
		return 0;
	}
	void add(const LayerParameters &new_gradient);
	void multiply(const double value);
	void reset();
};

#endif // LAYER_PARAMETERS
