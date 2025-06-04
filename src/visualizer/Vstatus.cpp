#include "Vstatus.hpp"
#include "fonts.hpp"

namespace Visualizer {
vStatus::vStatus(const std::shared_ptr<state> vstate_)
    : panel(vstate_),
      VRender({VSTATUS_WIDTH, VSTATUS_HEIGHT}) {}

void vStatus::clear() {
	VRender.clear(sf::Color::White);
}

void vStatus::do_render() {
	clear();
	drawText();
}

std::string vStatus::get_text() {
	std::ostringstream ss;
	ss << TextLabels::CURRENT_PHASE_TEXT << NNmodeName[(int)vstate->nnMode.load()] << std::endl
	   << TextLabels::RUNNING_MODE_TEXT << NNRunningModeName[vstate->settings.pause.load()] << std::endl
	   << TextLabels::ALGORITHM_MODE_TEXT << algorithmName[(int)vstate->AlgorithmMode.load()] << std::endl
	   << TextLabels::FPS_TEXT << fps << "/" << FPS_LIMIT << std::endl
	   << TextLabels::CURRENT_BATCH_TEXT << vstate->current_batch << "/" << vstate->config.training_config.batch_count << std::endl
	   << TextLabels::BATCH_SIZE_TEXT << vstate->config.training_config.batch_size << std::endl
	   << TextLabels::LERNING_RATE_TEXT << vstate->config.training_config.learning_rate << std::endl;
	return ss.str();
}

void vStatus::drawText() {
	sf::Text text(Fonts::getFont());
	text.setCharacterSize(STATUS_TEXT_FONT);
	text.setString(get_text());
	text.setFillColor(sf::Color::Black);
	VRender.draw(text);
}

void vStatus::display() {
	VRender.display();
}

sf::Sprite vStatus::getSprite() {
	display();
	return sf::Sprite(VRender.getTexture());
}

void vStatus::update_fps(const float fps_) {
	fps = fps_;
	set_update();
}
} // namespace Visualizer
