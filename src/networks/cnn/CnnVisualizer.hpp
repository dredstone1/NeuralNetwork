#ifndef CNNVISUALNETWORK
#define CNNVISUALNETWORK

#include <network/IvisualNetwork.hpp>

namespace nn::visualizer::cnn {
class CnnVisualier : public IVisualNetwork {
  private:
	const model::cnn::CNNConfig &config;

	void renderNetwork() override;
	void renderLayers();
	void renderLayer(const int index);

  public:
	CnnVisualier(
	    const std::shared_ptr<StateManager> state_,
	    const std::uint32_t width,
	    const model::cnn::CNNConfig &_config);
	~CnnVisualier() = default;

	void initLayer(const int index);
};
} // namespace nn::visualizer::cnn

#endif // CNNVISUALNETWORK
