#include "VisualizerController.hpp"

namespace nn::visualizer {
VisualManager::VisualManager(const model::ConfigData &_config) : config(_config) {
	printf("start Visualizer\n");
}

void VisualManager::initState() {
	if (!Vstate) {
		return;
	}

	auto &modes = config.visualizer_config.modes;

	for (size_t i = 0; i < modes.size(); i++) {
		Vstate->setState(modes[i].state, modes[i].mode);
	}
}

VisualManager::~VisualManager() { stop(); }

void VisualManager::stop() {
	running = false;

	if (display_thread.joinable()) {
		if (renderer) {
			renderer->close();
		}

		display_thread.join();
	}
}

void VisualManager::start() {
	if (renderer) {
		return;
	}

	display_thread = std::thread(&VisualManager::start_visuals, this);
}

void VisualManager::start_visuals() {
	Vstate = std::make_shared<StateManager>(config);
	if (!Vstate) {
		return;
	}

	initState();

	renderer = std::make_unique<VisualRender>(Vstate);
	if (!renderer) {
		return;
	}

	running.store(true);
	renderer->start();

	running.store(false);
}

void VisualManager::updateDots(const int layer, const model::Neurons &newNeurons) {
	if (!checkPointers()) {
		return;
	}

	renderer->updateDots(layer, newNeurons);
}

void VisualManager::update(const int layer, const model::LayerParameters &gradient_) {
	if (!checkPointers()) {
		return;
	}

	renderer->update(layer, gradient_);
}

void VisualManager::setNewPhaseMode(const NnMode nn_mode) {
	if (!checkPointers()) {
		return;
	}

	renderer->setNewPhaseMode(nn_mode);
}

void VisualManager::update(const training::gradient &new_grad) {
	if (!checkPointers()) {
		return;
	}

	renderer->update(new_grad);
}

void VisualManager::updateBatchCounter(const int batch) {
	if (!checkPointers()) {
		return;
	}

	Vstate->currentBatch = batch;
}

void VisualManager::updateError(const global::ValueType error, const int index) {
	if (!checkPointers()) {
		return;
	}

	renderer->updateBatchCounter(error, index);
}

void VisualManager::updateAlgoritemMode(const AlgorithmMode algoritem_mode) {
	if (!checkPointers()) {
		return;
	}

	Vstate->algorithmMode = algoritem_mode;
	Vstate->settings.exitTraining = algoritem_mode == AlgorithmMode::Normal;
}

void VisualManager::updatePrediction(const int index) {
	if (!checkPointers()) {
		return;
	}

	renderer->updatePrediction(index);
}

void VisualManager::updateLearningRate(const global::ValueType newLerningRate) {
	if (!checkPointers()) {
		return;
	}

	renderer->updateLearningRate(newLerningRate);
}

bool VisualManager::exit_training() {
	if (!checkPointers()) {
		return false;
	}

	return Vstate->getState(SettingType::ExitTraining);
}
} // namespace nn::visualizer
