#include "CnnVisualizer.hpp"

namespace nn::visualizer::cnn {
CnnVisualier::CnnVisualier(
    const std::shared_ptr<StateManager> state_,
    const std::uint32_t width,
    const model::CNNConfig &_config)
    : IVisualNetwork(state_, width),
      config(_config) {
}

void CnnVisualier::renderNetwork() {
	renderLayers();
}

void CnnVisualier::renderLayers() {
}

void CnnVisualier::renderLayer(const int index) {
}

void CnnVisualier::initLayer(const int index) {
}
} // namespace nn::visualizer::cnn
