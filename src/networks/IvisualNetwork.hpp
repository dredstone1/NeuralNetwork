#ifndef IVISUALNETWORK
#define IVISUALNETWORK

#include "../visualizer/panel.hpp"
#include <SFML/Graphics.hpp>

namespace nn::visualizer {
constexpr std::uint32_t MODEL_HEIGHT = 770u;
constexpr std::uint32_t MODEL_WIDTH = 1055u;
constexpr std::uint32_t NEURON_WIDTH = 40;
constexpr float SUB_NETWORKS_WIDTH = MODEL_WIDTH - NEURON_WIDTH * 2;


constexpr sf::Color MODEL_BG = PANELS_BG;

class IVisualNetwork : public Panel {
  private:
	void clear();
	void display();

	void doRender() override;

  protected:
	const float visualWidth;
	sf::RenderTexture networkRender;

	virtual void renderNetwork() = 0;

  public:
	IVisualNetwork(
	    const std::shared_ptr<StateManager> state_,
	    const std::uint32_t width);
	virtual ~IVisualNetwork() = default;

	sf::Sprite getSprite() { return sf::Sprite(networkRender.getTexture()); }
};
} // namespace nn::visualizer

#endif // IVISUALNETWORK
