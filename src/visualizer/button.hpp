#ifndef BUTTON
#define BUTTON

#include "panel.hpp"
#include <SFML/Graphics.hpp>
#include <string>
#include <string_view>

namespace nn::visualizer {
constexpr std::uint32_t BUTTON_HEIGHT = 50;
constexpr std::uint32_t BUTTON_WIDTH = 200;
constexpr std::uint32_t BUTTON_TEXT_FONT = 30;

namespace buttoncolors {
constexpr sf::Color ACTIVE(0, 123, 255);
constexpr sf::Color INACTIVE(187, 187, 187);
constexpr sf::Color TEXT_ACTIVE(255, 255, 255);
constexpr sf::Color TEXT_INACTIVE(34, 34, 34);
} // namespace buttoncolors

class Button : public Panel {
  private:
	sf::RenderTexture buttonRender;
	const SettingType CurrentState;
	bool visibleState;
	const std::string lable;

	void sendCommand();
	void renderButton();
	void display();
	void drawText();
	sf::Color getBgColor();
	sf::Color getFontColor();
	void doRender() override;
	void observe() override;

  public:
	Button(const std::shared_ptr<StateManager> _state, const std::string_view &lable, const SettingType state_);
	~Button() = default;
	sf::Sprite getSprite();
	bool checkForClick(const sf::Vector2f mousePos, const sf::Vector2f boxPos);
};
} // namespace nn::visualizer

#endif // BUTTON
