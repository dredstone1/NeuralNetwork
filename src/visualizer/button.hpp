#ifndef BUTTON
#define BUTTON

#include "state.hpp"
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/System/Vector2.hpp>

namespace Visualizer {
#define BUTTON_HEIGHT 50
#define BUTTON_WIDTH 200
#define BUTTON_TEXT_FONT 30

class button {
  private:
	sf::RenderTexture buttonRender;
	state &State;
	const states CurrentState;
	bool visibleState;
	const std::string lable;
	void sendCommand();
	void renderButton();
	void display();
	void drawText();
	sf::Color getBgColor();

  public:
	button(state &_state, const std::string lable, const states state_);
	~button() = default;
	sf::Sprite getSprite();
	bool checkForClick(const sf::Vector2f mousePos, sf::Vector2f boxPos);
	void render();
};
} // namespace Visualizer

#endif // VISUALIZERCONTROLLER
