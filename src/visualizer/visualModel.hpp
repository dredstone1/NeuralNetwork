#ifndef VISUALMODEL
#define VISUALMODEL

#include "panel.hpp"
#include <Globals.hpp>
#include <SFML/Graphics.hpp>

namespace nn::visualizer {
constexpr std::uint32_t MODEL_HEIGHT = 770u;
constexpr std::uint32_t MODEL_WIDTH = 1055u;

constexpr sf::Color MODEL_BG = PANELS_BG;

class ModelPanel : public Panel {
  private:
	void clear();
	void display();
	void createVInterface();

	void doRender() override;

  protected:
	sf::RenderTexture modelRender;

  public:
	ModelPanel(const std::shared_ptr<StateManager> state_);
	virtual ~ModelPanel() = default;

	sf::Sprite getSprite();
};
} // namespace nn::visualizer

#endif // VISUALMODEL
