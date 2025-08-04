#ifndef IVISUALNETWORK
#define IVISUALNETWORK

#include "../../src/visualizer/panel.hpp"
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/System/Vector2.hpp>

namespace nn::visualizer {
constexpr std::uint32_t MODEL_HEIGHT = 770u;
constexpr std::uint32_t MODEL_WIDTH = 1055u;
constexpr float SUB_NETWORKS_WIDTH = MODEL_WIDTH - global::NEURON_WIDTH * 2;

constexpr sf::Color MODEL_BG = PANELS_BG;

class IVisualNetwork : public Panel {
  private:
	void clear() { networkRender.clear(sf::Color::Red); }
	void display() { networkRender.display(); }
	void doRender() override;

	bool shouldSleep() const;

  protected:
	float visualWidth;
	sf::RenderTexture networkRender;

	virtual void renderNetwork() = 0;

  public:
	IVisualNetwork(
	    const std::shared_ptr<StateManager> state_,
	    const std::uint32_t width)
	    : Panel(state_),
	      visualWidth(width),
	      networkRender({width, MODEL_HEIGHT}) {}
	virtual ~IVisualNetwork() = default;

	sf::Sprite getSprite() { return sf::Sprite(networkRender.getTexture()); }
	void setVstate(std::shared_ptr<StateManager> state_) { vstate = state_; }
	void attempPause();

	virtual void setWidth(const std::uint32_t newWidth) {
		visualWidth = newWidth;
		if (networkRender.resize({newWidth, networkRender.getSize().y})) {
		}
	}
};
} // namespace nn::visualizer

#endif // IVISUALNETWORK
