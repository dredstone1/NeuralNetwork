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
	ss << TextLabels::CURRENT_PHASE_TEXT << NNmodeName[(int)vstate->nnMode.load()] << std::endl
	   << TextLabels::RUNNING_MODE_TEXT << NNRunningModeName[vstate->settings.pause.load()] << std::endl
	   << TextLabels::ALGORITHM_MODE_TEXT << algorithmName[(int)vstate->algorithmMode.load()] << std::endl
	   << TextLabels::FPS_TEXT << fps << "/" << FPS_LIMIT << std::endl
	   << TextLabels::CURRENT_BATCH_TEXT << vstate->currentBatch << "/" << vstate->config.training_config.batch_count << "#" << batchPerSecond << std::endl
	   << TextLabels::BATCH_SIZE_TEXT << vstate->config.training_config.batch_size << std::endl
	   << TextLabels::LERNING_RATE_TEXT << learningRate << std::endl;
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
