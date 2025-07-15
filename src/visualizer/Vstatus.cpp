#include "Vstatus.hpp"
#include "fonts.hpp"

namespace nn::visualizer {
StatusPanel::StatusPanel(const std::shared_ptr<StateManager> vstate_)
    : Panel(vstate_),
      VRender({VSTATUS_WIDTH, VSTATUS_HEIGHT}) {}

void StatusPanel::clear() {
	VRender.clear(STATUSE_PANEL_COLOR);
}

void StatusPanel::doRender() {
	clear();
	drawText();
	display();
}

std::string StatusPanel::getText() {
	static float avgBatchPerSecond = 0.0f;
	static int updatesCount = 0;

	updatesCount++;
	avgBatchPerSecond += (batchPerSecond - avgBatchPerSecond) / updatesCount;

	std::ostringstream ss;

	const int totalBatches = vstate->config.trainingConfig.getBatchCount();
	const int currentBatch = vstate->currentBatch;
	const int remainingBatches = totalBatches - currentBatch;

	float minutesLeft = 0.0f;
	if (avgBatchPerSecond > 0.01f) {
		minutesLeft = remainingBatches / avgBatchPerSecond / 60.0f;
	}

	ss << TextLabels::CURRENT_PHASE_TEXT << NNmodeName[(int)vstate->nnMode.load()] << "\n"
	   << TextLabels::RUNNING_MODE_TEXT << NNRunningModeName[vstate->settings.pause.load()] << "\n"
	   << TextLabels::ALGORITHM_MODE_TEXT << algorithmName[(int)vstate->algorithmMode.load()] << "\n"
	   << TextLabels::FPS_TEXT << fps << "/" << FPS_LIMIT << "\n"
	   << TextLabels::CURRENT_BATCH_TEXT << currentBatch << "/" << totalBatches << " #" << batchPerSecond << "\n"
	   << TextLabels::BATCH_SIZE_TEXT << vstate->config.trainingConfig.getBatchSize() << "\n"
	   << TextLabels::LEARNING_RATE_TEXT << learningRate << "\n"
	   << "Time Left: " << std::fixed << std::setprecision(2) << minutesLeft << " min\n"
	   << "Time Left: " << std::fixed << std::setprecision(2) << minutesLeft / 60 << " hours";

	return ss.str();
}

void StatusPanel::drawText() {
	sf::Text text(Fonts::getFont());
	text.setCharacterSize(STATUS_TEXT_FONT);
	text.setString(getText());
	text.setFillColor(TEXT_COLOR);

	VRender.draw(text);
}

void StatusPanel::display() {
	VRender.display();
}

sf::Sprite StatusPanel::getSprite() {
	return sf::Sprite(VRender.getTexture());
}

void StatusPanel::updateFps(const float newFps) {
	fps = newFps;
	setUpdate();
}

void StatusPanel::updateBps(const float newBps) {
	batchPerSecond = newBps;
	setUpdate();
}

void StatusPanel::updateLerningRate(const global::ValueType newLerningRate) {
	learningRate = newLerningRate;
	setUpdate();
}
} // namespace nn::visualizer
