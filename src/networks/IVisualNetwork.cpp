#include <network/IvisualNetwork.hpp>
#include <thread>

namespace nn::visualizer {
void IVisualNetwork::doRender() {
	clear();
	renderNetwork();
	display();
}

bool IVisualNetwork::shouldSleep() const {
	return vstate->settings.pause ||
	       (vstate->settings.preciseMode && updateStatus());
}

void IVisualNetwork::attempPause() {
	if (!vstate) {
		return;
	}

	if (vstate->settings.autoPause.load()) {
		vstate->settings.pause = true;
	}

	while (shouldSleep()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
}
} // namespace nn::visualizer
