#ifndef VINTERFACE
#define VINTERFACE

#include "button.hpp"

namespace nn::visualizer {

// ============================================================================
// VISUAL INTERFACE CONSTANTS
// ============================================================================

/// Width of the visual interface panel in pixels
constexpr std::uint32_t VINTERFACE_WIDTH = 500;

/// Gap between buttons in pixels
constexpr std::uint32_t BUTTON_GAP = 10;

/// Number of buttons per column
constexpr int BUTTON_PER_COLLUM = 3;

/// Height of the visual interface panel in pixels
constexpr std::uint32_t VINTERFACE_HEIGHT = BUTTON_HEIGHT * BUTTON_PER_COLLUM + BUTTON_GAP * (BUTTON_PER_COLLUM - 1);

/// Background color for the interface panel
constexpr sf::Color INTERFACE_PANEL_COLOR = PANELS_BG;

/**
 * @class InterfacePanel
 * @brief Visual interface panel for neural network controls
 * 
 * This class provides a graphical interface panel that contains buttons
 * and controls for interacting with the neural network visualization.
 * It handles user input events and manages the visual state of the interface.
 * 
 * @section features Key Features
 * - **Interactive Buttons**: Various control buttons for visualization features
 * - **Event Handling**: Mouse click and keyboard input processing
 * - **Rendering**: Real-time rendering of the interface panel
 * - **State Management**: Integration with the visualization state system
 * 
 * @section usage Usage Example
 * ```cpp
 * auto stateManager = std::make_shared<StateManager>();
 * InterfacePanel panel(stateManager);
 * 
 * // Handle user input
 * panel.handleClick(mousePosition, panelPosition);
 * 
 * // Get rendered sprite for display
 * sf::Sprite sprite = panel.getSprite();
 * ```
 */
class InterfacePanel : public Panel {
  private:
	sf::RenderTexture VRender;  ///< Render texture for the interface panel
	bool needHandlePress{false};  ///< Flag indicating if press handling is needed
	std::vector<std::unique_ptr<Button>> buttons;  ///< Collection of interface buttons

	/**
	 * @brief Creates the visual interface layout
	 * 
	 * Initializes and positions all buttons and UI elements
	 * in the interface panel.
	 */
	void createVInterface();
	
	/**
	 * @brief Displays the interface panel
	 * 
	 * Renders the interface panel to the render texture.
	 */
	void display();
	
	/**
	 * @brief Handles key press events
	 * @param mousePos_ Current mouse position
	 * @param boxPos Position of the interface panel
	 */
	void handleKeyPresed(const sf::Vector2i mousePos_, const sf::Vector2f boxPos);
	
	/**
	 * @brief Performs the actual rendering
	 * 
	 * @note This method is called by the base Panel class
	 */
	void doRender() override;

  public:
	/**
	 * @brief Constructs an interface panel
	 * @param vstate Shared pointer to the state manager
	 */
	InterfacePanel(const std::shared_ptr<StateManager> vstate);
	
	/**
	 * @brief Destructor
	 */
	~InterfacePanel() = default;

	/**
	 * @brief Gets the rendered sprite of the interface panel
	 * @return SFML sprite containing the rendered interface
	 */
	sf::Sprite getSprite();

	/**
	 * @brief Handles mouse click events
	 * @param mousePos_ Position of the mouse click
	 * @param boxPos Position of the interface panel
	 * 
	 * @note This method processes button clicks and updates the interface state
	 */
	void handleClick(const sf::Vector2i mousePos_, const sf::Vector2f boxPos);
	
	/**
	 * @brief Handles the case when no click is detected
	 * 
	 * @note This method is called when the mouse is not clicked
	 * @note Used for updating button states and visual feedback
	 */
	void handleNoClick();
};
} // namespace nn::visualizer

#endif // VINTERFACE
