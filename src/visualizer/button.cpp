/**
 * @file button.cpp
 * @brief Implementation of interactive button components for the neural network visualizer
 * 
 * This file implements UI button functionality for the visualization interface,
 * providing interactive controls for toggling various visualization features.
 * Buttons handle mouse interactions, visual state changes, and command dispatch
 * to the state management system.
 */

#include "button.hpp"
#include "fonts.hpp"

namespace nn::visualizer {

/**
 * @brief Constructs a new Button with specified label and state binding
 * 
 * Creates an interactive button that can toggle a specific setting in the
 * state manager. The button renders with different colors based on its
 * current state (active/inactive).
 * 
 * @param _state Shared pointer to the state manager for this visualizer session
 * @param lable Text label to display on the button
 * @param initState The setting type this button controls (e.g., graph visibility)
 */
Button::Button(
    const std::shared_ptr<StateManager> _state,
    const std::string_view &lable,
    const SettingType initState)
    : Panel(_state),
      buttonRender({BUTTON_WIDTH, BUTTON_HEIGHT}),
      CurrentState(initState),
      lable(lable) {
	renderButton();
}

/**
 * @brief Renders the button with current state-appropriate appearance
 * 
 * Updates the button's visual appearance by clearing the render texture with
 * the appropriate background color and redrawing the text. This method is
 * called whenever the button's state changes.
 */
void Button::renderButton() {
	buttonRender.clear(getBgColor());
	drawText();
	visibleState = vstate->getState(CurrentState);
}

/**
 * @brief Determines the background color based on button's current state
 * 
 * @return Active color if the associated setting is enabled, inactive color otherwise
 */
sf::Color Button::getBgColor() {
	if (vstate->getState(CurrentState)) {
		return buttoncolors::ACTIVE;
	} else {
		return buttoncolors::INACTIVE;
	}
}

/**
 * @brief Determines the text color based on button's current state
 * 
 * @return Active text color if the associated setting is enabled, inactive text color otherwise
 */
sf::Color Button::getFontColor() {
	if (vstate->getState(CurrentState)) {
		return buttoncolors::TEXT_ACTIVE;
	} else {
		return buttoncolors::TEXT_INACTIVE;
	}
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
