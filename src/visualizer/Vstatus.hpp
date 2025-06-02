#ifndef VSTATUS
#define VSTATUS

#include "panel.hpp"
#include "state.hpp"
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/RenderTexture.hpp>
#include <SFML/System/Vector2.hpp>
#include <climits>
#include <curses.h>
#include <memory>

namespace Visualizer {
#define VSTATUS_WIDTH 500
#define VSTATUS_HEIGHT 585
#define STATUS_TEXT_FONT 30
#define CURRENT_PHASE_TEXT "current phase: "
#define RUNNING_MODE_TEXT "running mode: "
#define FPS_TEXT "fps: "
#define FPS_LIMIT 60

const std::string NNRunningModeName[] = {
    "Running",
    "Pause",
};

class vStatus : public panel {
  private:
	sf::RenderTexture VRender;
	void createVstatus();
	void display();
	void drawText();
	void clear();
	std::string get_text();
	float fps;
	void do_render() override;

  public:
	vStatus(std::shared_ptr<state> vstate_);
	sf::Sprite getSprite();
	void update_fps(const float fps);
};
} // namespace Visualizer

#endif // VSTATUS
