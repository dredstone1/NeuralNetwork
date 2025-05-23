#include "VisualizerController.hpp"
#include "VisualizerRenderer.hpp"
#include <cstdio>
#include <thread>

namespace Visualizer {
visualizerController::visualizerController(const neural_network &network) {
	printf("start Visualizer\n");
	start(network);
}

visualizerController::~visualizerController() {
	stop();
}

void visualizerController::stop() {
	running = false;
	if (display_thread.joinable()) {
		if (renderer) {
			renderer->close();
		}
		display_thread.join();
	}
}

void visualizerController::start(const neural_network &network) {
	if (running)
		return;

	display_thread = thread(&visualizerController::start_visuals, this, cref(network));
	while (!renderer) {
		this_thread::sleep_for(1ms);
	}
}

void visualizerController::start_visuals(const neural_network &network) {
	Vstate = new state;
	if (!Vstate)
		return;

	renderer = new VisualizerRenderer(network, Vstate);
	if (!renderer)
		return;

	running = true;
	renderer->start();
	delete renderer;
	delete Vstate;
	running = false;
}

void visualizerController::wait_until_updated() {
	if (!renderer || !Vstate->preciseMode)
		return;

	while (renderer->updateStatus()) {
		this_thread::sleep_for(1ms);
	}
}

void visualizerController::pause() {
	if (!renderer || !Vstate)
		return;

	while (Vstate->pause) {
		this_thread::sleep_for(10ms);
	}
}

void visualizerController::autoPause() {
	if (Vstate->autoPause)
		Vstate->pause = true;
	pause();
}

void visualizerController::update(const neural_network &network) {
	if (renderer) {
	}
}

void visualizerController::updateDots(const int layer, vector<double> out, vector<double> net) {
	if (renderer) {
		renderer->updateDots(layer, out, net);
		wait_until_updated();
		pause();
		autoPause();
	}
}

void visualizerController::update(const int layer, const LayerParameters &gradient_) {
	if (renderer) {
		renderer->update(layer, gradient_);
		wait_until_updated();
		pause();
		autoPause();
	}
}
} // namespace Visualizer
