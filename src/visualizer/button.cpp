#include "button.hpp"
#include "fonts.hpp"

namespace nn::visualizer {
Button::Button(const std::shared_ptr<StateManager> _state, const std::string_view &lable, const SettingType initState)
    : Panel(_state),
      buttonRender({BUTTON_WIDTH, BUTTON_HEIGHT}),
      CurrentState(initState),
      lable(lable) {
	renderButton();
}

void Button::renderButton() {
	buttonRender.clear(getBgColor());
	drawText();
	visibleState = vstate->getState(CurrentState);
}

sf::Color Button::getBgColor() {
	if (vstate->getState(CurrentState))
		return buttoncolors::ACTIVE;
	else
		return buttoncolors::INACTIVE;
}

sf::Color Button::getFontColor() {
	if (vstate->getState(CurrentState))
		return buttoncolors::TEXT_ACTIVE;
	else
		return buttoncolors::TEXT_INACTIVE;
}

void Button::drawText() {
	std::ostringstream ss;
	sf::Text text(Fonts::getFont());
	text.setCharacterSize(BUTTON_TEXT_FONT);
	text.setString(lable);
	text.setFillColor(getFontColor());
	text.setPosition({2, 2});
	buttonRender.draw(text);
}

void Button::display() {
	buttonRender.display();
}

void Button::sendCommand() {
	vstate->toggle(CurrentState);
}

sf::Sprite Button::getSprite() {
	return sf::Sprite(buttonRender.getTexture());
}

bool Button::checkForClick(const sf::Vector2f mousePos, const sf::Vector2f boxPos) {
	sf::Sprite button_box = getSprite();
	button_box.setPosition(boxPos);
	if (button_box.getGlobalBounds().contains(mousePos)) {
		setUpdate();
		sendCommand();
		return true;
	}

	return false;
}

void Button::doRender() {
	if (vstate->getState(CurrentState) != visibleState) {
		renderButton();
	}
	display();
}

void Button::observe() {
	if (vstate->getState(CurrentState) != visibleState) {
		setUpdate();
	}
}
} // namespace nn::visualizer
