#ifndef BUTTON
#define BUTTON

#include "panel.hpp"
#include "state.hpp"
#include <SFML/Graphics.hpp>
#include <memory>

namespace Visualizer {
constexpr const int BUTTON_HEIGHT = 50;
constexpr const int BUTTON_WIDTH = 200;
constexpr const int BUTTON_TEXT_FONT = 30;

namespace ButtonColors {
const inline sf::Color ACTIVE = sf::Color(0, 123, 255);
const inline sf::Color INACTIVE = sf::Color(187, 187, 187);
const inline sf::Color TEXT_ACTIVE = sf::Color::White;
const inline sf::Color TEXT_INACTIVE = sf::Color(34, 34, 34);
} // namespace ButtonColors

class button : public panel {
  private:
	sf::RenderTexture buttonRender;
	const states CurrentState;
	bool visibleState;
	const std::string lable;
	void sendCommand();
	void renderButton();
	void display();
	void drawText();
	sf::Color getBgColor();
	sf::Color getFontColor();
	void do_render() override;

  public:
	button(const std::shared_ptr<state> _state, const std::string lable, const states state_);
	~button() = default;
	sf::Sprite getSprite();
	bool checkForClick(const sf::Vector2f mousePos, const sf::Vector2f boxPos);
};
} // namespace Visualizer

#endif // BUTTON
