#include "CnnVisualizer.hpp"

namespace nn::visualizer::cnn {
CnnVisualier::CnnVisualier(
    const std::shared_ptr<StateManager> state_,
    const std::uint32_t width,
    const model::cnn::CNNConfig &_config)
    : IVisualNetwork(state_, width),
      config(_config) {
}

void CnnVisualier::renderNetwork() {
}
} // namespace nn::visualizer::cnn
