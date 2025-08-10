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

	int hours = 0, minutes = 0, seconds = 0;
	if (avgBatchPerSecond > 0.01f) {
		float totalSeconds = remainingBatches / avgBatchPerSecond;
		int totalSec = static_cast<int>(totalSeconds);
		hours = totalSec / 3600;
		minutes = (totalSec % 3600) / 60;
		seconds = totalSec % 60;
	}

	ss << TextLabels::CURRENT_PHASE_TEXT << NNmodeName[(int)vstate->nnMode.load()] << "\n";
	if (vstate->config.visualConfig.enableNetwrokVisual) {
		ss << TextLabels::RUNNING_MODE_TEXT << NNRunningModeName[vstate->settings.pause.load()] << "\n";
	}
	ss << TextLabels::ALGORITHM_MODE_TEXT << algorithmName[(int)vstate->algorithmMode.load()] << "\n"
	   << TextLabels::FPS_TEXT << fps << "/" << FPS_LIMIT << "\n"
	   << TextLabels::CURRENT_BATCH_TEXT << currentBatch << "/" << totalBatches << " #" << batchPerSecond << "\n"
	   << TextLabels::BATCH_SIZE_TEXT << vstate->config.trainingConfig.getBatchSize() << "\n"
	   << TextLabels::LEARNING_RATE_TEXT << learningRate << "\n"
	   << "Time Left: "
	   << std::setw(2) << std::setfill('0') << hours << ":"
	   << std::setw(2) << std::setfill('0') << minutes << ":"
	   << std::setw(2) << std::setfill('0') << seconds;

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
