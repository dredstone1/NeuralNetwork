#include "button.hpp"
#include "fonts.hpp"
#include "state.hpp"
#include <cmath>
#include <memory>
#include <sstream>

namespace Visualizer {
button::button(const std::shared_ptr<state> _state, const std::string lable, const states Cstate)
    : panel(_state),
      CurrentState(Cstate),
      lable(lable) {
	renderButton();
}

void button::renderButton() {
	buttonRender.create(BUTTON_WIDTH, BUTTON_HEIGHT);
	buttonRender.clear(getBgColor());
	drawText();
	visibleState = vstate->getState(CurrentState);
}

sf::Color button::getBgColor() {
	if (vstate->getState(CurrentState))
		return ButtonColors::ACTIVE;
	else
		return ButtonColors::INACTIVE;
}

sf::Color button::getFontColor() {
	if (vstate->getState(CurrentState))
		return ButtonColors::TEXT_ACTIVE;
	else
		return ButtonColors::TEXT_INACTIVE;
}

void button::drawText() {
	std::ostringstream ss;
	sf::Text text;
	text.setFont(Fonts::getFont());
	text.setCharacterSize(BUTTON_TEXT_FONT);
	text.setString(lable);
	text.setFillColor(getFontColor());
	text.setPosition(2, 2);
	buttonRender.draw(text);
}

void button::display() {
	buttonRender.display();
}

void button::sendCommand() {
	vstate->toggle(CurrentState);
}

sf::Sprite button::getSprite() {
	display();
	return sf::Sprite(buttonRender.getTexture());
}

bool button::checkForClick(const sf::Vector2f mousePos, const sf::Vector2f boxPos) {
	sf::Sprite button_box = getSprite();
	button_box.setPosition(boxPos);
	if (button_box.getGlobalBounds().contains(mousePos)) {
		set_update();
		sendCommand();
		return true;
	}

	return false;
}

void button::do_render() {
	if (vstate->getState(CurrentState) != visibleState) {
		renderButton();
	}
}
} // namespace Visualizer
