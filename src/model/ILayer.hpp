#ifndef LAYER
#define LAYER

#include <Globals.hpp>

namespace nn::model {
class ILayer {
  public:
	ILayer();
	virtual ~ILayer() = default;

	virtual void forward(const global::ParamMetrix &metrix);
	virtual void updateWeight(const global::ParamMetrix &metrix);
	virtual void backword(const global::ParamMetrix &metrix);
};

} // namespace nn::model

#endif // LAYER
