#include "VInterface.hpp"
#include "button.hpp"
#include "state.hpp"

namespace nn {
namespace Visualizer {
vInteface::vInteface(const std::shared_ptr<state> vstate)
    : panel(vstate),
      VRender({VINTERFACE_WIDTH, VINTERFACE_HEIGHT}) {
	createVInterface();
}

void vInteface::createVInterface() {
	VRender.clear(INTERFACE_PANEL_COLOR);

	buttons.reserve(STATES_COUNT);

	for (int i = 0; i < STATES_COUNT; i++) {
		buttons.push_back(std::make_unique<button>(vstate, vstate->getStateString((states)i), (states)i));
	}
}

void vInteface::display() {
	VRender.display();
}

sf::Sprite vInteface::getSprite() {
	return sf::Sprite(VRender.getTexture());
}

void vInteface::handleClick(const sf::Vector2i mousePos_, const sf::Vector2f boxPos) {
	if (!needHandlePress) {
		set_update();
		needHandlePress = true;
		handleKeyPresed(mousePos_, boxPos);
	}
}

void vInteface::handleNoClick() {
	needHandlePress = false;
	set_update();
}

void vInteface::do_render() {
	int row = 0;
	int column = -1;

	for (size_t button_ = 0; button_ < buttons.size(); button_++) {
		if (button_ % BUTTON_PER_COLLUM == 0) {
			row = 0;
			column++;
		} else {
			row++;
		}
		buttons[button_]->render();
		sf::Sprite buttonSprite = buttons[button_]->getSprite();
		buttonSprite.setPosition(sf::Vector2f((BUTTON_WIDTH + BUTTON_GAP) * column, row * (BUTTON_HEIGHT + BUTTON_GAP)));
		VRender.draw(buttonSprite);
	}

	display();
}

void vInteface::handleKeyPresed(const sf::Vector2i mousePos_, const sf::Vector2f boxPos) {
	sf::Vector2f mousePos(static_cast<float>(mousePos_.x), static_cast<float>(mousePos_.y));

	int row = 0;
	int column = -1;
	for (size_t button_ = 0; button_ < buttons.size(); button_++) {
		if (button_ % BUTTON_PER_COLLUM == 0) {
			row = 0;
			column++;
		} else {
			row++;
		}
		if (buttons[button_]->checkForClick(mousePos, {boxPos.x + (BUTTON_WIDTH + BUTTON_GAP) * column, boxPos.y + (BUTTON_HEIGHT + BUTTON_GAP) * row})) {
			return;
		}
	}
}
} // namespace Visualizer
} // namespace nn
