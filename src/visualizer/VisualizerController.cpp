#include "VisualizerController.hpp"
#include "tensor.hpp"

namespace nn::visualizer {
VisualManager::VisualManager(const model::Config &_config) : config(_config) {
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
	printf("start Visualizer\n");
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

void VisualManager::updateLoss(
    const global::ValueType newDataLoss,
    int index) {
	if (!checkPointers()) {
		return;
	}

	renderer->updateLoss(newDataLoss, index);
}

void VisualManager::updateEvaluate(
    const global::ValueType newDataEvaluate, int index) {
	if (!checkPointers()) {
		return;
	}

	renderer->updateEvaluate(newDataEvaluate, index);
}

void VisualManager::updateAlgorithmMode(const AlgorithmMode algoritem_mode) {
	if (!checkPointers()) {
		return;
	}

	Vstate->algorithmMode = algoritem_mode;
}


void VisualManager::updatePrediction(const global::Tensor &out) {
	if (!checkPointers()) {
		return;
	}

	renderer->updatePrediction(out);
}

void VisualManager::updateInput(const global::Tensor &input) {
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

void VisualManager::addVisualSubNetwork(
    const std::shared_ptr<IVisualNetwork> newVisual) {
	if (!checkPointers()) {
		return;
	}

	renderer->addVisualSubNetwork(newVisual);
}

void VisualManager::resetGraph() {
	if (!checkPointers()) {
		return;
	}

    renderer->resetGraph();
}

} // namespace nn::visualizer
