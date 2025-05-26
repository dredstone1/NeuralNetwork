#ifndef VSTATUS
#define VSTATUS

#include "state.hpp"
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/RenderTexture.hpp>
#include <SFML/System/Vector2.hpp>
#include <climits>

using namespace std;

namespace Visualizer {
#define VSTATUS_WIDTH 500
#define VSTATUS_HEIGHT 585
#define STATUS_TEXT_FONT 30
#define CURRENT_PHASE_TEXT "current phase: "

class vStatus {
  private:
	sf::RenderTexture VRender;
	state *vstate;
	void createVstatus();
	void display();
	void drawText();
	void clear();
	string get_text();

  public:
	vStatus(state *vstate_);
	~vStatus() = default;
	sf::Sprite getSprite();
	void renderStatus();
};
} // namespace Visualizer
#endif
