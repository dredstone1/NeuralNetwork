#ifndef VINTERFACE
#define VINTERFACE

#include "button.hpp"
#include "state.hpp"
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/RenderTexture.hpp>
#include <SFML/System/Vector2.hpp>
#include <climits>

using namespace std;

namespace Visualizer {
#define VINTERFACE_WIDTH 500
#define VINTERFACE_HEIGHT 170
#define BUTTON_GAP 10

class vInteface {
  private:
	sf::RenderTexture VRender;
	state *vstate;
	void createVInterface();
	void display();
	void handleKeyPresed(sf::Vector2i mousePos_, sf::Vector2f boxPos);
	bool needHandlePress{false};
	vector<button *> buttons;

  public:
	vInteface(state *vstate);
	~vInteface() = default;
	sf::Sprite getSprite();
	void handleClick(sf::Vector2i mousePos_, sf::Vector2f boxPos);
	void renderInterface();
	void handleNoClick();
};
} // namespace Visualizer
#endif
