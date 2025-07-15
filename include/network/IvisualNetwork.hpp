#ifndef IVISUALNETWORK
#define IVISUALNETWORK

#include "../src/visualizer/panel.hpp"
#include <SFML/Graphics.hpp>
#include <thread>

namespace nn::visualizer {
constexpr std::uint32_t MODEL_HEIGHT = 770u;
constexpr std::uint32_t MODEL_WIDTH = 1055u;
constexpr float SUB_NETWORKS_WIDTH = MODEL_WIDTH - global::NEURON_WIDTH * 2;

constexpr sf::Color MODEL_BG = PANELS_BG;

class IVisualNetwork : public Panel {
  private:
	void clear() { networkRender.clear(MODEL_BG); }
	void display() { networkRender.display(); }

	void doRender() override {
		clear();

		renderNetwork();

		display();
	}

  protected:
	const float visualWidth;
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

	void setVstate(std::shared_ptr<StateManager> state_) {
		vstate = state_;
	}

	void attempPause() {
		if (!vstate) {
			return;
		}

		if (vstate->settings.autoPause.load()) {
			vstate->settings.pause = true;
		}

		while (vstate->settings.pause) {
			std::this_thread::sleep_for(std::chrono::milliseconds(500));
		}
		while (vstate->settings.preciseMode && updateStatus()) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}
};
} // namespace nn::visualizer

#endif // IVISUALNETWORK
