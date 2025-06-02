#include "Vstatus.hpp"
#include "fonts.hpp"
#include "visualizer/panel.hpp"
#include "visualizer/state.hpp"
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <cmath>
#include <memory>
#include <sstream>

namespace Visualizer {
vStatus::vStatus(std::shared_ptr<state> vstate_) : panel(vstate_) {
	createVstatus();
}

void vStatus::clear() {
	VRender.clear(sf::Color::White);
}

void vStatus::createVstatus() {
	VRender.create(VSTATUS_WIDTH, VSTATUS_HEIGHT);
	clear();
}

void vStatus::do_render() {
	clear();
	drawText();
}

std::string vStatus::get_text() {
	std::ostringstream ss;
	ss << CURRENT_PHASE_TEXT << NNmodeName[(int)vstate->nnMode.load()] << std::endl
	   << RUNNING_MODE_TEXT << NNRunningModeName[vstate->pause.load()] << std::endl
	   << FPS_TEXT << fps << std::endl;
	return ss.str();
}

void vStatus::drawText() {
	sf::Text text;
	text.setFont(Fonts::getFont());
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
	need_update = true;
}
} // namespace Visualizer
