#include "VisualizerController.hpp"
#include "VisualizerRenderer.hpp"
#include "model/config.hpp"
#include <cstddef>
#include <cstdio>
#include <thread>

namespace Visualizer {
visualizerController::visualizerController(VisualizerConfig &_config) : renderer(NULL), config(_config) {
	printf("start Visualizer\n");
}

void visualizerController::initState() {
	if (!Vstate)
		return;

	for (size_t i = 0; i < config.modes.size(); i++) {
		Vstate->setState(config.modes[i].state, config.modes[i].mode);
	}
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

	initState();

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

	while (renderer->updateStatus() && running) {
		this_thread::sleep_for(1ms);
	}
}

void visualizerController::pause() {
	if (!renderer || !Vstate)
		return;

	while (Vstate->pause && running) {
		this_thread::sleep_for(10ms);
	}
}

void visualizerController::autoPause() {
	if (Vstate->autoPause)
		Vstate->pause = true;
	pause();
}

void visualizerController::handleStates() {
	if (!running)
		return;

	wait_until_updated();
	pause();
	autoPause();
}

bool visualizerController::checkP() {
	return (renderer && Vstate);
}

void visualizerController::updateDots(const int layer, vector<double> out, vector<double> net) {
	if (checkP()) {
		if (!running) {
			stop();
			return;
		}

		renderer->updateDots(layer, out, net);
		handleStates();
	}
}

void visualizerController::update(const int layer, const LayerParameters &gradient_) {
	if (checkP()) {
		if (!running) {
			stop();
			return;
		}

		renderer->update(layer, gradient_);
		handleStates();
	}
}
} // namespace Visualizer
