#ifndef VSTATUS
#define VSTATUS

#include "panel.hpp"
#include <SFML/Graphics.hpp>

namespace nn::visualizer {
constexpr std::uint32_t VSTATUS_WIDTH = 500;
constexpr std::uint32_t VSTATUS_HEIGHT = 255;
constexpr int STATUS_TEXT_FONT = 20;
constexpr int FPS_LIMIT = 144;

namespace TextLabels {
constexpr std::string_view CURRENT_PHASE_TEXT = "current phase: ";
constexpr std::string_view RUNNING_MODE_TEXT = "running mode: ";
constexpr std::string_view ALGORITHM_MODE_TEXT = "current algorithm mode: ";
constexpr std::string_view CURRENT_BATCH_TEXT = "current batch: ";
constexpr std::string_view BATCH_SIZE_TEXT = "batch size: ";
constexpr std::string_view LERNING_RATE_TEXT = "learning rate: ";
constexpr std::string_view FPS_TEXT = "fps: ";
} // namespace TextLabels

constexpr std::array<std::string_view, 2> NNRunningModeName = {"Running", "Pause"};
constexpr std::array<std::string_view, 2> algorithmName = {"Normal", "Training"};
constexpr std::array<std::string_view, 2> NNmodeName = {"Forword", "Backward"};

constexpr sf::Color TEXT_COLOR(0, 0, 0);
constexpr sf::Color STATUSE_PANEL_COLOR = PANELS_BG;

class StatusPanel : public Panel {
  private:
	sf::RenderTexture VRender;
	float fps;
	float batchPerSecond;
	global::ValueType learningRate;

	std::string getText();
	void display();
	void drawText();
	void clear();
	void doRender() override;

  public:
	StatusPanel(const std::shared_ptr<StateManager> vstate_);
	~StatusPanel() = default;
	sf::Sprite getSprite();
	void updateFps(const float fps);
	void updateBps(const float bps);
	void updateLerningRate(const global::ValueType newLarningRate);
};
} // namespace nn::visualizer

#endif // VSTATUS
