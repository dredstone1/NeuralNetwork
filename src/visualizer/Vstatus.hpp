#ifndef VSTATUS
#define VSTATUS

#include "Globals.hpp"
#include "panel.hpp"
#include "state.hpp"
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Color.hpp>

namespace nn {
namespace Visualizer {
constexpr std::uint32_t VSTATUS_WIDTH = 500;
constexpr std::uint32_t VSTATUS_HEIGHT = 255;
constexpr int STATUS_TEXT_FONT = 20;
constexpr int FPS_LIMIT = 144;

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

constexpr sf::Color TEXT_COLOR(0, 0, 0);
constexpr sf::Color STATUSE_PANEL_COLOR = PANELS_BG;

class vStatus : public panel {
  private:
	sf::RenderTexture VRender;
	void display();
	void drawText();
	void clear();
	std::string get_text();
	float fps;
	float batchPerSecond;
    Global::ValueType lr;
	void do_render() override;

  public:
	vStatus(const std::shared_ptr<state> vstate_);
	sf::Sprite getSprite();
	void update_fps(const float fps);
	void update_bps(const float bps);
	void update_lr(const Global::ValueType lr_);
};
} // namespace Visualizer
} // namespace nn

#endif // VSTATUS
