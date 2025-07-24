#ifndef OPTIMIZERS
#define OPTIMIZERS

#include "config.hpp"
#include <Globals.hpp>
#include <cmath>

namespace nn::model {

class IOptimizer {
  protected:
	int batchSize;

  public:
	virtual ~IOptimizer() = default;

	virtual void step(global::ValueType *weight, const global::ValueType *grad, std::size_t size) = 0;
	virtual void reset() = 0;

	void setOfset(const int batchSize_) { batchSize = batchSize_; }
};

class ConstantOptimizer : public IOptimizer {
  private:
	const ConstantOptimizerConfig &config;

  public:
	ConstantOptimizer(const ConstantOptimizerConfig &config_) : config(config_) {}

	void step(global::ValueType *weight, const global::ValueType *grad, std::size_t size) override {
		for (std::size_t i = 0; i < size; ++i) {
			weight[i] -= config.getLearningRate() * grad[i] / static_cast<global::ValueType>(batchSize);
		}
	}

	void reset() override {}
};

class AdamOptimizer : public IOptimizer {
  private:
	const AdamOptimizerConfig &config;
	std::vector<global::ValueType> m; // 1st moment
	std::vector<global::ValueType> v; // 2nd moment
	std::size_t timestep = 0;

  public:
	AdamOptimizer(const AdamOptimizerConfig &config_) : config(config_) {}

	void step(global::ValueType *weight, const global::ValueType *grad, std::size_t size) override {
		if (m.size() != size) {
			m.resize(size, 0.0f);
			v.resize(size, 0.0f);
		}

		++timestep;

		const auto lr = config.getLearningRate();
		const auto beta1 = config.getBeta1();
		const auto beta2 = config.getBeta2();
		const auto epsilon = config.getEpsilon();

		for (std::size_t i = 0; i < size; ++i) {
			const global::ValueType g = grad[i]; // ❗ Don't divide again if already averaged

			m[i] = beta1 * m[i] + (1.0f - beta1) * g;
			v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

			const global::ValueType m_hat = m[i] / (1.0f - std::pow(beta1, timestep));
			const global::ValueType v_hat = v[i] / (1.0f - std::pow(beta2, timestep));

			weight[i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
		}
	}

	void reset() override {
		std::fill(m.begin(), m.end(), 0);
		std::fill(v.begin(), v.end(), 0);
		timestep = 0;
	}
};

} // namespace nn::model

#endif // OPTIMIZERS
