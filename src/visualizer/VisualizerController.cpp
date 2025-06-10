#include "VisualizerController.hpp"

namespace nn::visualizer {
visualizerController::visualizerController(const model::ConfigData &_config)
    : config(_config) {
	printf("start Visualizer\n");
}

void visualizerController::initState() {
	if (!Vstate)
		return;

	for (size_t i = 0; i < config.visualizer_config.modes.size(); i++) {
		Vstate->setState(config.visualizer_config.modes[i].state, config.visualizer_config.modes[i].mode);
	}
}

visualizerController::~visualizerController() {
	stop();
}

void visualizerController::stop() {
	running.store(false);
	if (display_thread.joinable()) {
		if (renderer) {
			renderer->close();
		}

		display_thread.join();
	}
}

void visualizerController::start(const model::NeuralNetwork &network) {
	if (running.load())
		return;

	display_thread = std::thread(&visualizerController::start_visuals, this, std::cref(network));
	wait_until_started();
}

void visualizerController::start_visuals(const model::NeuralNetwork &network) {
	Vstate = std::make_shared<StateManager>(config);
	if (!Vstate)
		return;

	initState();

	renderer = std::make_unique<VisualizerRenderer>(network, Vstate);
	if (!renderer) {
		return;
	}

	running.store(true);
	renderer->start();

	running.store(false);
}

void visualizerController::wait_until_started() {
	while (true) {
		if (renderer && Vstate)
			break;

		std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_DELAY_MM));
	}
}

void visualizerController::wait_until_updated() {
	if (!renderer || !Vstate->settings.preciseMode)
		return;

	while (renderer->updateStatus() && running.load()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_DELAY_MM));
	}
}

void visualizerController::pause() {
	if (!renderer || !Vstate)
		return;

	while (Vstate->settings.pause.load() && running.load()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_DELAY_MM));
	}
}

void visualizerController::autoPause() {
	if (Vstate->settings.autoPause.load())
		Vstate->settings.pause.store(true);

	pause();
}

void visualizerController::handleStates() {
	if (!running.load())
		return;

	wait_until_updated();
	pause();
	autoPause();
}

bool visualizerController::checkP() {
	return (renderer && Vstate);
}

void visualizerController::updateDots(const int layer, const model::Neurons &newNeurons) {
	if (checkP()) {
		if (!running.load()) {
			stop();
			return;
		}

		// for debuging
		// for (size_t i = 0; i < net.size(); i++) {
		// 	if (net[i] > 100 ||std::isnan(net[i]) || !std::isfinite(net[i])) {
		// 		pause();
		// 		Vstate->settings.pause.store(true);
		// 		Vstate->settings.autoPause.store(true);
		// 		Vstate->settings.preciseMode.store(true);
		// 		printf("work, dot: %f\n", net[i]);
		// 		handleStates();
		// 		return;
		// 	}
		// }

		handleStates();
		renderer->updateDots(layer, newNeurons);
	}
}

void visualizerController::update(const int layer, const model::LayerParameters &gradient_) {
	if (checkP()) {
		if (!running.load()) {
			stop();
			return;
		}

		// for debuging
		// for (size_t i = 0; i < gradient_.weights.size(); i++) {
		// 	for (size_t j = 0; j < gradient_.weights[i].size(); j++) {
		// 		if (gradient_.weights[i][j] > 100 || std::isnan(gradient_.weights[i][j]) || !std::isfinite(gradient_.weights[i][j])) {
		// 			pause();
		// 			Vstate->settings.pause.store(true);
		// 			Vstate->settings.preciseMode.store(true);
		// 			Vstate->settings.autoPause.store(true);
		// 			printf("work, weight: %f\n", gradient_.weights[i][j]);
		// 			handleStates();
		// 			return;
		// 		}
		// 	}
		// }

		handleStates();

		renderer->update(layer, gradient_);
	}
}

void visualizerController::setNewPhaseMode(const NnMode nn_mode) {
	if (checkP()) {
		if (!running.load()) {
			stop();
			return;
		}

		wait_until_updated();
		renderer->setNewPhaseMode(nn_mode);
	}
}

void visualizerController::update(const training::gradient &new_grad) {
	if (checkP()) {
		if (!running.load()) {
			stop();
			return;
		}

		handleStates();
		renderer->update(new_grad);
	}
}

void visualizerController::updateBatchCounter(const int batch) {
	if (checkP()) {
		if (!running.load()) {
			stop();
			return;
		}

		wait_until_updated();
		Vstate->currentBatch = batch;
	}
}

void visualizerController::updateError(const global::ValueType error, const int index) {
	if (checkP()) {
		if (!running.load()) {
			stop();
			return;
		}

		wait_until_updated();
		renderer->updateBatchCounter(error, index);
	}
}

void visualizerController::updateAlgoritemMode(const AlgorithmMode algoritem_mode) {
	if (checkP()) {
		if (!running.load()) {
			stop();
			return;
		}

		wait_until_updated();
		Vstate->algorithmMode = algoritem_mode;
		Vstate->settings.exitTraining = algoritem_mode == AlgorithmMode::Normal;
	}
}

void visualizerController::update_prediction(const int index) {
	if (checkP()) {
		if (!running.load()) {
			stop();
			return;
		}

		wait_until_updated();
		renderer->update_prediction(index);
	}
}

void visualizerController::update_lr(const global::ValueType newLerningRate) {
	if (checkP()) {
		if (!running.load()) {
			stop();
			return;
		}

		wait_until_updated();
		renderer->update_lr(newLerningRate);
	}
}

bool visualizerController::exit_training() {
	return Vstate->getState(SettingType::ExitTraining);
}
} // namespace nn::visualizer
