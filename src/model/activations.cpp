#include "activations.hpp"

namespace ActivationFunctions {
// double Relu::activation(const double x) const {
//     return max(0.0, x);
// }
//
// double Relu::derivative(const double x) const {
//     return (x > 0) ? 1.0 : 0.0;
// }
//
// double LeakyRelu::activation(const double x) const {
//     return (x > 0) ? x : RELU_LEAKY_ALPHA * x;
// }
//
// double LeakyRelu::derivative(const double x) const {
//     return (x > 0) ? 1.0 : RELU_LEAKY_ALPHA;
// }
//
// double Softmax::max_vector(const vector<double> &metrix) {
//     double max = metrix[0];
//
//     for (auto &value : metrix) {
//         if (value > max) {
//             max = value;
//         }
//     }
//
//     return max;
// }
//
// void Softmax::softmax(neurons &metrix) {
//     double max = max_vector(metrix.net);
//     double sum = 0.0;
//
//     for (int i = 0; i < metrix.size(); ++i) {
//         metrix.out[i] = exp(metrix.net[i] - max);
//         sum += metrix.out[i];
//     }
//
//     for (int i = 0; i < metrix.size(); ++i) {
//         metrix.out[i] /= sum;
//     }
// }

static double max_vector(const vector<double> &metrix) {
	double max = metrix[0];
	for (auto &value : metrix) {
		if (value > max) {
			max = value;
		}
	}
	return max;
}

void Softmax(neurons &metrix) {
	double max = max_vector(metrix.net);
	double sum = 0.0;
	for (int i = 0; i < metrix.size(); ++i) {
		metrix.out[i] = exp(metrix.net[i] - max);
		sum += metrix.out[i];
	}
	for (int i = 0; i < metrix.size(); ++i) {
		metrix.out[i] /= sum;
	}
}
} // namespace ActivationFunctions
