#include "VInterface.hpp"

namespace nn::visualizer {
vInteface::vInteface(const std::shared_ptr<StateManager> vstate)
    : Panel(vstate),
      VRender({VINTERFACE_WIDTH, VINTERFACE_HEIGHT}) {
	createVInterface();
}

void vInteface::createVInterface() {
	VRender.clear(INTERFACE_PANEL_COLOR);

	buttons.reserve(STATES_COUNT);

	for (int i = 0; i < STATES_COUNT; i++) {
		buttons.push_back(std::make_unique<Button>(vstate, vstate->getStateString((SettingType)i), (SettingType)i));
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
		setUpdate();
		needHandlePress = true;
		handleKeyPresed(mousePos_, boxPos);
	}
}

void vInteface::handleNoClick() {
	needHandlePress = false;
	setUpdate();
}

void vInteface::doRender() {
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
} // namespace nn::visualizer
