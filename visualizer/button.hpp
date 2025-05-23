#ifndef BUTTON
#define BUTTON

#include "state.hpp"
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/System/Vector2.hpp>

namespace Visualizer {
#define BUTTON_HEIGHT 100
#define BUTTON_WIDTH 200

class button {
  private:
	sf::RenderTexture buttonRender;
	state *State;
	const states CurrentState;
	const string lable;
	void sendCommand();
	void createButton();
	void display();
	void drawText();

  public:
	button(state *_state, string lable, const states state_);
	~button() = default;
	sf::Sprite getSprite();
	bool checkForClick(sf::Vector2f mousePos, sf::Vector2f boxPos);
};
} // namespace Visualizer

#endif // VISUALIZERCONTROLLER
