#include "FnnVisualizer.hpp"
#include <SFML/Graphics/Color.hpp>

namespace nn::visualizer {
FnnVisualier::FnnVisualier(const std::shared_ptr<StateManager> state_,
                           const std::uint32_t width)
    : IVisualNetwork(state_, width) {
}

void FnnVisualier::renderNetwork() {
}

void FnnVisualier::createNetwork() {
}
} // namespace nn::visualizer
