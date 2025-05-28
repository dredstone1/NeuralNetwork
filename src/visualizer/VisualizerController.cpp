#include "VisualizerController.hpp"
#include "VisualizerRenderer.hpp"
#include "model/config.hpp"
#include "model/neuralNetwork.hpp"
#include "state.hpp"
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <thread>

namespace Visualizer {
visualizerController::visualizerController(const VisualizerConfig &_config) : renderer(NULL), config(_config) {
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
		wait_until_stop();
		display_thread.join();
	}

	delete renderer;
	delete Vstate;
}

void visualizerController::start(const neural_network &network) {
	if (running)
		return;

	display_thread = std::thread(&visualizerController::start_visuals, this, std::cref(network));
	wait_until_started();
}

void visualizerController::start_visuals(const neural_network &network) {
	Vstate = new state;
	if (!Vstate)
		return;

	initState();

	renderer = new VisualizerRenderer(network, *Vstate);
	if (!renderer) {
		delete Vstate;
		return;
	}

	running = true;
	renderer->start();

	delete renderer;
	renderer = NULL;
	delete Vstate;
	Vstate = NULL;

	running = false;
}

void visualizerController::wait_until_started() {
	while (!renderer || !Vstate) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
}

void visualizerController::wait_until_stop() {
	while (renderer && Vstate) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
}

void visualizerController::wait_until_updated() {
	if (!renderer || !Vstate->preciseMode)
		return;

	while (renderer->updateStatus() && running) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
}

void visualizerController::pause() {
	if (!renderer || !Vstate)
		return;

	while (Vstate->pause && running) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
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

void visualizerController::updateDots(const int layer, std::vector<double> out, std::vector<double> net) {
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

void visualizerController::setNewPhaseMode(const NNmode nn_mode) {
	if (checkP()) {
		if (!running) {
			stop();
			return;
		}

		renderer->setNewPhaseMode(nn_mode);
		wait_until_updated();
	}
}
} // namespace Visualizer
