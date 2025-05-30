#ifndef VINTERFACE
#define VINTERFACE

#include "button.hpp"
#include "state.hpp"
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/RenderTexture.hpp>
#include <SFML/System/Vector2.hpp>
#include <climits>

namespace Visualizer {
#define VINTERFACE_WIDTH 500
#define VINTERFACE_HEIGHT 170
#define BUTTON_GAP 10

class vInteface {
  private:
	sf::RenderTexture VRender;
	state &vstate;
	void createVInterface();
	void display();
	void handleKeyPresed(const sf::Vector2i mousePos_, const sf::Vector2f boxPos);
	bool needHandlePress{false};
	std::vector<button *> buttons;

  public:
	vInteface(state &vstate);
	~vInteface();
	sf::Sprite getSprite();
	void handleClick(const sf::Vector2i mousePos_, const sf::Vector2f boxPos);
	void renderInterface();
	void handleNoClick();
};
} // namespace Visualizer

#endif // VINTERFACE
