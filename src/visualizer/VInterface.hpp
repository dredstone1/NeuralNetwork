#ifndef VINTERFACE
#define VINTERFACE

#include "button.hpp"

namespace nn::visualizer {
constexpr std::uint32_t VINTERFACE_WIDTH = 500;
constexpr std::uint32_t BUTTON_GAP = 10;
constexpr int BUTTON_PER_COLLUM = 3;
constexpr std::uint32_t VINTERFACE_HEIGHT = BUTTON_HEIGHT * BUTTON_PER_COLLUM + BUTTON_GAP * (BUTTON_PER_COLLUM - 1);

constexpr sf::Color INTERFACE_PANEL_COLOR = PANELS_BG;

class IntefacePanel : public Panel {
  private:
	sf::RenderTexture VRender;
	bool needHandlePress{false};
	std::vector<std::unique_ptr<Button>> buttons;

	void createVInterface();
	void display();
	void handleKeyPresed(const sf::Vector2i mousePos_, const sf::Vector2f boxPos);
	void doRender() override;

  public:
	IntefacePanel(const std::shared_ptr<StateManager> vstate);
	~IntefacePanel() = default;

	sf::Sprite getSprite();

	void handleClick(const sf::Vector2i mousePos_, const sf::Vector2f boxPos);
	void handleNoClick();
};
} // namespace nn::visualizer
#endif // VINTERFACE
