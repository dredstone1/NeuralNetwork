#ifndef OPTIMIZERS
#define OPTIMIZERS

#include "config.hpp"
#include "tensor.hpp"
#include <cmath>

namespace nn::model {
class IOptimizer {
  protected:
	int batchSize;

  public:
	virtual ~IOptimizer() = default;

	virtual void step(global::Tensor &weight, const global::Tensor &grad) = 0;
	virtual void reset() = 0;

	void setOfset(const int batchSize_) { batchSize = batchSize_; }
};

class ConstantOptimizer : public IOptimizer {
  private:
	const ConstantOptimizerConfig &config;

  public:
	ConstantOptimizer(const ConstantOptimizerConfig &config_)
	    : config(config_) {}

	void step(global::Tensor &weight, const global::Tensor &grad) override;

	void reset() override {}
};
} // namespace nn::model

#endif // OPTIMIZERS
