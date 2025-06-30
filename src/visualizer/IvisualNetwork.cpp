#include "IvisualNetwork.hpp"

namespace nn::visualizer {
IVisualNetwork::IVisualNetwork(
    const std::shared_ptr<StateManager> state_,
    const std::uint32_t width)
    : Panel(state_),
      visualWidth(width),
      networkRender({width, MODEL_HEIGHT}) {
}

void IVisualNetwork::clear() {
	networkRender.clear(MODEL_BG);
}

void IVisualNetwork::display() {
	networkRender.display();
}

void IVisualNetwork::doRender() {
	clear();

	renderNetwork();

	display();
}
} // namespace nn::visualizer
