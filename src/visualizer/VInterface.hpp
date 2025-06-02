#ifndef VINTERFACE
#define VINTERFACE

#include "button.hpp"
#include "panel.hpp"
#include "state.hpp"
#include <SFML/Graphics.hpp>
#include <climits>
#include <memory>

namespace Visualizer {
constexpr const int VINTERFACE_WIDTH = 500;
constexpr const int VINTERFACE_HEIGHT = 170;
constexpr const int BUTTON_GAP = 10;

class vInteface : public panel {
  private:
	sf::RenderTexture VRender;
	void createVInterface();
	void display();
	void handleKeyPresed(const sf::Vector2i mousePos_, const sf::Vector2f boxPos);
	bool needHandlePress{false};
	std::vector<std::unique_ptr<button>> buttons;
	void do_render() override;

  public:
	vInteface(const std::shared_ptr<state> vstate);
	~vInteface() = default;
	sf::Sprite getSprite();
	void handleClick(const sf::Vector2i mousePos_, const sf::Vector2f boxPos);
	void handleNoClick();
};
} // namespace Visualizer

#endif // VINTERFACE
