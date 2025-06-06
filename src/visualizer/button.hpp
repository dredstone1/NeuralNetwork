#ifndef BUTTON
#define BUTTON

#include "panel.hpp"
#include "state.hpp"
#include <SFML/Graphics.hpp>

namespace nn {
namespace Visualizer {
constexpr std::uint32_t BUTTON_HEIGHT = 50;
constexpr std::uint32_t BUTTON_WIDTH = 200;
constexpr std::uint32_t BUTTON_TEXT_FONT = 30;

namespace ButtonColors {
constexpr sf::Color ACTIVE(0, 123, 255);
constexpr sf::Color INACTIVE(187, 187, 187);
constexpr sf::Color TEXT_ACTIVE(255, 255, 255);
constexpr sf::Color TEXT_INACTIVE(34, 34, 34);
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
	void observe() override;

  public:
	button(const std::shared_ptr<state> _state, const std::string lable, const states state_);
	~button() = default;
	sf::Sprite getSprite();
	bool checkForClick(const sf::Vector2f mousePos, const sf::Vector2f boxPos);
};
} // namespace Visualizer
} // namespace nn

#endif // BUTTON
