#include "VInterface.hpp"
#include "button.hpp"
#include "state.hpp"
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/System/Vector2.hpp>
#include <cstddef>
#include <memory>

namespace Visualizer {
vInteface::vInteface(std::shared_ptr<state> vstate)
    : panel(vstate) {
	createVInterface();
}

void vInteface::createVInterface() {
	VRender.create(VINTERFACE_WIDTH, VINTERFACE_HEIGHT);
	VRender.clear(sf::Color::Magenta);

	buttons.reserve(STATES_COUNT);

	for (int i = 0; i < STATES_COUNT; i++) {
		buttons.push_back(new button(vstate, vstate->getStateString((states)i), (states)i));
	}
}

void vInteface::display() {
	VRender.display();
}

sf::Sprite vInteface::getSprite() {
	display();
	return sf::Sprite(VRender.getTexture());
}

void vInteface::handleClick(const sf::Vector2i mousePos_, const sf::Vector2f boxPos) {
	if (!needHandlePress) {
        need_update = true;
		needHandlePress = true;
		handleKeyPresed(mousePos_, boxPos);
	}
}

void vInteface::handleNoClick() {
	needHandlePress = false;
    need_update = true;
}

void vInteface::do_render() {
	for (size_t button_ = 0; button_ < buttons.size(); button_++) {
		buttons[button_]->render();
		sf::Sprite buttonSprite = sf::Sprite(buttons[button_]->getSprite());
		buttonSprite.setPosition(0, button_ * (BUTTON_HEIGHT + BUTTON_GAP));
		VRender.draw(buttonSprite);
	}
}

void vInteface::handleKeyPresed(const sf::Vector2i mousePos_, const sf::Vector2f boxPos) {
	sf::Vector2f mousePos(static_cast<float>(mousePos_.x), static_cast<float>(mousePos_.y));
	for (size_t button_ = 0; button_ < buttons.size(); button_++) {
		if (buttons[button_]->checkForClick(mousePos, {boxPos.x, boxPos.y + (BUTTON_HEIGHT + 10) * button_}))
			return;
	}
}

vInteface::~vInteface() {
	for (size_t i = 0; i < buttons.size(); i++) {
		delete buttons[i];
	}
}
} // namespace Visualizer
