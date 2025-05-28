#include "button.hpp"
#include "fonts.hpp"
#include "state.hpp"
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/System/Vector2.hpp>
#include <cmath>
#include <sstream>

namespace Visualizer {
button::button(state &_state, const std::string lable, const states Cstate) : State(_state), CurrentState(Cstate), lable(lable) {
	renderButton();
}

void button::renderButton() {
	buttonRender.create(BUTTON_WIDTH, BUTTON_HEIGHT);
	buttonRender.clear(getBgColor());
	drawText();
	visibleState = State.getState(CurrentState);
}

sf::Color button::getBgColor() {
	if (State.getState(CurrentState))
		return sf::Color::Red;
	else
		return sf::Color::Yellow;
}

void button::drawText() {
	std::ostringstream ss;
	sf::Text text;
	text.setFont(Fonts::getFont());
	text.setCharacterSize(BUTTON_TEXT_FONT);
	text.setString(lable);
	text.setFillColor(sf::Color::Black);
	text.setPosition(2, 2);
	buttonRender.draw(text);
}

void button::display() {
	buttonRender.display();
}

void button::sendCommand() {
	State.toggle(CurrentState);
}

sf::Sprite button::getSprite() {
	display();
	return sf::Sprite(buttonRender.getTexture());
}

bool button::checkForClick(sf::Vector2f mousePos, sf::Vector2f boxPos) {
	sf::Sprite button_box = getSprite();
	button_box.setPosition(boxPos);
	if (button_box.getGlobalBounds().contains(mousePos)) {
		sendCommand();
		return true;
	}

	return false;
}

void button::render() {
	if (State.getState(CurrentState) != visibleState) {
		renderButton();
	}
}
} // namespace Visualizer
