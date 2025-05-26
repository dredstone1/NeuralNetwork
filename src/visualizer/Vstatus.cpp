#include "Vstatus.hpp"
#include "fonts.hpp"
#include "visualizer/state.hpp"
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <cmath>
#include <sstream>

namespace Visualizer {
vStatus::vStatus(state *vstate_) : vstate(vstate_) {
	createVstatus();
}

void vStatus::clear() {
	VRender.clear(sf::Color::Magenta);
}

void vStatus::createVstatus() {
	VRender.create(VSTATUS_WIDTH, VSTATUS_HEIGHT);
	clear();
}

void vStatus::renderStatus() {
	clear();
	drawText();
}

void vStatus::drawText() {
	ostringstream ss;
	ss << NNmodeName[(int)vstate->nnMode.load()] << endl;
	sf::Text text;
	text.setFont(Fonts::getFont());
	text.setCharacterSize(30);
	text.setString(ss.str());
	text.setFillColor(sf::Color::Black);
	text.setPosition(2, 2);
	VRender.draw(text);
}

void vStatus::display() {
	VRender.display();
}

sf::Sprite vStatus::getSprite() {
	display();
	return sf::Sprite(VRender.getTexture());
}
} // namespace Visualizer
