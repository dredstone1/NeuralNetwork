#ifndef IVISUALNETWORK
#define IVISUALNETWORK

#include "../../src/visualizer/panel.hpp"
#include <Globals.hpp>
#include <SFML/Graphics.hpp>

namespace nn::visualizer {

/**
 * @brief Default height of the visual network panel.
 */
constexpr std::uint32_t MODEL_HEIGHT = 770u;

/**
 * @brief Default width of the visual network panel.
 */
constexpr std::uint32_t MODEL_WIDTH = 1055u;

/**
 * @brief Width available for sub-networks, taking into account neuron widths.
 */
constexpr float SUB_NETWORKS_WIDTH = MODEL_WIDTH - global::NEURON_WIDTH * 2;

/**
 * @brief Background color of the network panel.
 */
constexpr sf::Color MODEL_BG = PANELS_BG;

/**
 * @class IVisualNetwork
 * @brief Abstract base class for visualizing neural networks.
 *
 * This class extends the Panel class and provides a framework for rendering
 * a neural network using SFML. It manages the off-screen render texture,
 * background clearing, and display. Derived classes must implement the
 * `renderNetwork()` function to define the actual drawing behavior.
 */
class IVisualNetwork : public Panel {
  private:
	/**
	 * @brief Clears the render texture using the panel background color.
	 */
	void clear() { networkRender.clear(PANELS_BG); }

	/**
	 * @brief Displays the content of the render texture to the screen.
	 */
	void display() { networkRender.display(); }

	/**
	 * @brief Internal rendering function called by the panel loop.
	 * Clears, renders, and displays the network.
	 */
	void doRender() override;

	/**
	 * @brief Determines if the visual network should pause rendering.
	 * @return true if the network should pause, false otherwise.
	 */
	bool shouldSleep() const;

  protected:
	/** Width of the visual network panel. */
	float visualWidth;

	/** Render texture used for off-screen drawing of the network. */
	sf::RenderTexture networkRender;

	/**
	 * @brief Pure virtual function that must be implemented by derived classes
	 * to render the network content.
	 */
	virtual void renderNetwork() = 0;

  public:
	/**
	 * @brief Constructs the visual network panel.
	 * @param state_ Shared pointer to the state manager.
	 * @param width Width of the visual network panel.
	 */
	IVisualNetwork(
	    const std::shared_ptr<StateManager> state_,
	    const std::uint32_t width)
	    : Panel(state_),
	      visualWidth(width),
	      networkRender({width, MODEL_HEIGHT}) {}

	/** @brief Virtual destructor. */
	virtual ~IVisualNetwork() = default;

	/**
	 * @brief Returns the network render texture as a sprite.
	 * @return SFML sprite of the rendered network.
	 */
	sf::Sprite getSprite() { return sf::Sprite(networkRender.getTexture()); }

	/**
	 * @brief Sets the current visual state manager.
	 * @param state_ Shared pointer to the state manager.
	 */
	void setVstate(std::shared_ptr<StateManager> state_) { vstate = state_; }

	/**
	 * @brief Attempts to pause rendering based on the state manager settings.
	 * This function will sleep the thread while the network should pause.
	 */
	void attempPause();

	/**
	 * @brief Sets a new width for the visual network panel and resizes the render texture.
	 * @param newWidth New width in pixels.
	 */
	virtual void setWidth(const std::uint32_t newWidth);
};

} // namespace nn::visualizer

#endif // IVISUALNETWORK
