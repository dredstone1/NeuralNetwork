#ifndef VSTATUS
#define VSTATUS

#include "panel.hpp"
#include "state.hpp"
#include <SFML/Graphics.hpp>
#include <climits>
#include <curses.h>
#include <memory>

namespace Visualizer {
constexpr const int VSTATUS_WIDTH = 500;
constexpr const int VSTATUS_HEIGHT = 585;
constexpr const int STATUS_TEXT_FONT = 30;
constexpr const int FPS_LIMIT = 144;

namespace TextLabels {
constexpr const char *CURRENT_PHASE_TEXT = "current phase: ";
constexpr const char *RUNNING_MODE_TEXT = "running mode: ";
constexpr const char *ALGORITHM_MODE_TEXT = "current algorithm mode: ";
constexpr const char *CURRENT_BATCH_TEXT = "current batch: ";
constexpr const char *BATCH_SIZE_TEXT = "batch size: ";
constexpr const char *LERNING_RATE_TEXT = "learning rate: ";
constexpr const char *FPS_TEXT = "fps: ";
} // namespace TextLabels

const std::array<std::string, 2> NNRunningModeName = {"Running", "Pause"};
const std::array<std::string, 2> algorithmName = {"Normal", "Training"};
const std::array<std::string, 2> NNmodeName = {"Forword", "Backward"};

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
