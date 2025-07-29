#include "VisualizerController.hpp"

namespace nn::visualizer {
VisualManager::VisualManager(const model::Config &_config) : config(_config) {
	printf("start Visualizer\n");
}

void VisualManager::initState() {
	if (!Vstate) {
		return;
	}

	auto &modes = config.visualConfig.modes;

	for (size_t i = 0; i < modes.size(); ++i) {
		Vstate->setState(modes[i].state, modes[i].mode);
	}
}

VisualManager::~VisualManager() { stop(); }

void VisualManager::stop() {
	running = false;

	if (displayThread.joinable()) {
		if (renderer) {
			renderer->close();
		}

		displayThread.join();
	}
}

void VisualManager::start() {
	if (renderer) {
		return;
	}

	displayThread = std::thread(&VisualManager::startVisuals, this);
	while (!checkPointers()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
}

void VisualManager::startVisuals() {
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

void VisualManager::setNewPhaseMode(const NnMode nn_mode) {
	if (!checkPointers()) {
		return;
	}

	renderer->setNewPhaseMode(nn_mode);
}

void VisualManager::updateBatchCounter(const int batch) {
	if (!checkPointers()) {
		return;
	}

	Vstate->currentBatch = batch;
}

void VisualManager::updateError(
    const global::ValueType newDataEvaluate,
    const global::ValueType newDataLost,
    int index) {
	if (!checkPointers()) {
		return;
	}

	renderer->updateBatchCounter(newDataEvaluate, newDataLost, index);
}

void VisualManager::updateAlgorithmMode(const AlgorithmMode algoritem_mode) {
	if (!checkPointers()) {
		return;
	}

	Vstate->algorithmMode = algoritem_mode;
}

void VisualManager::updatePrediction(const global::Prediction &pre) {
	if (!checkPointers()) {
		return;
	}

	renderer->updatePrediction(pre);
}

void VisualManager::updateInput(const global::ParamMetrix &input) {
	if (!checkPointers()) {
		return;
	}

	renderer->updateInput(input);
}

void VisualManager::updateLearningRate(const global::ValueType newLerningRate) {
	if (!checkPointers()) {
		return;
	}

	renderer->updateLearningRate(newLerningRate);
}

bool VisualManager::exitTraining() {
	if (!checkPointers()) {
		return false;
	}

	return Vstate->getState(SettingType::ExitTraining);
}

void VisualManager::addVisualSubNetwork(const std::shared_ptr<IVisualNetwork> newVisual) {
	if (!checkPointers()) {
		return;
	}

	renderer->addVisualSubNetwork(newVisual);
}
} // namespace nn::visualizer
