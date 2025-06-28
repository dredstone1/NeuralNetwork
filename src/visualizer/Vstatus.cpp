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
	std::ostringstream ss;
	ss << TextLabels::CURRENT_PHASE_TEXT << NNmodeName[(int)vstate->nnMode.load()] << "\n"
	   << TextLabels::RUNNING_MODE_TEXT << NNRunningModeName[vstate->settings.pause.load()] << "\n"
	   << TextLabels::ALGORITHM_MODE_TEXT << algorithmName[(int)vstate->algorithmMode.load()] << "\n"
	   << TextLabels::FPS_TEXT << fps << "/" << FPS_LIMIT << "\n"
	   << TextLabels::CURRENT_BATCH_TEXT << vstate->currentBatch << "/" << vstate->config.trainingConfig.batch_count << "#" << batchPerSecond << "\n"
	   << TextLabels::BATCH_SIZE_TEXT << vstate->config.trainingConfig.batch_size << "\n"
	   << TextLabels::LEARNING_RATE_TEXT << learningRate << "\n";
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
