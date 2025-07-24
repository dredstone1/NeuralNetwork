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

	virtual global::ValueType calculate(const global::ValueType value, const global::ValueType grad) = 0;

	virtual void reset() = 0;

	void setOfset(const int ofset_) { batchSize = ofset_; }
};

class ConstantOptimizer : public IOptimizer {
  private:
	const ConstantOptimizerConfig &config;

  public:
	ConstantOptimizer(const ConstantOptimizerConfig &config_) : config(config_) {}
	~ConstantOptimizer() = default;

	global::ValueType calculate(const global::ValueType, const global::ValueType grad);

	void reset() {}
};

} // namespace nn::model

#endif // OPTIMIZERS
