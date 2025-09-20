/**
 * @file button.hpp
 * @brief Header file for interactive button components in the neural network visualizer
 * 
 * This file defines the Button class and related constants for creating interactive
 * UI elements in the visualization interface. Buttons provide visual feedback and
 * allow users to toggle various visualization features during training.
 */

#ifndef BUTTON
#define BUTTON

#include "panel.hpp"
#include <SFML/Graphics.hpp>

namespace nn::visualizer {

// Button dimension constants
constexpr std::uint32_t BUTTON_HEIGHT = 50;   ///< Standard button height in pixels
constexpr std::uint32_t BUTTON_WIDTH = 200;   ///< Standard button width in pixels  
constexpr std::uint32_t BUTTON_TEXT_FONT = 30; ///< Font size for button text

/**
 * @namespace buttoncolors
 * @brief Color scheme constants for button visual states
 * 
 * Defines the color palette used for button backgrounds and text
 * in different interaction states (active/inactive).
 */
namespace buttoncolors {
constexpr sf::Color ACTIVE(0, 123, 255);        ///< Active button background color (blue)
constexpr sf::Color INACTIVE(187, 187, 187);    ///< Inactive button background color (gray)
constexpr sf::Color TEXT_ACTIVE(255, 255, 255); ///< Active button text color (white)
constexpr sf::Color TEXT_INACTIVE(34, 34, 34);  ///< Inactive button text color (dark gray)
} // namespace buttoncolors

/**
 * @class Button
 * @brief Interactive button component for the neural network visualizer
 * 
 * The Button class provides a clickable UI element that can toggle settings
 * in the visualizer state manager. It handles mouse interactions, visual state
 * changes, and provides visual feedback to the user about the current state
 * of the associated setting.
 * 
 * Key features:
 * - Visual state indication (active/inactive colors)
 * - Mouse click detection and handling
 * - Integration with state management system
 * - Automatic visual updates when state changes
 */
class Button : public Panel {
  private:
	sf::RenderTexture buttonRender;   ///< Render texture for button graphics
	const SettingType CurrentState;   ///< Setting type this button controls
	bool visibleState;                ///< Cached visual state for change detection
	const std::string lable;          ///< Text label displayed on the button

	/** @brief Sends toggle command to the state manager */
	void sendCommand();
	
	/** @brief Renders the button with current state colors and text */
	void renderButton();
	
	/** @brief Displays the rendered button to its render texture */
	void display();
	
	/** @brief Draws the button's text label with appropriate color */
	void drawText();
	
	/** @brief Gets background color based on current state */
	sf::Color getBgColor();
	
	/** @brief Gets text color based on current state */
	sf::Color getFontColor();
	
	/** @brief Overridden render method from Panel base class */
	void doRender() override;
	
	/** @brief Overridden observer method from Panel base class */
	void observe() override;

  public:
	/**
	 * @brief Constructs a button with specified label and state binding
	 * @param _state Shared state manager for the visualizer session
	 * @param lable Text to display on the button
	 * @param state_ Setting type this button will control
	 */
	Button(const std::shared_ptr<StateManager> _state, const std::string_view &lable, const SettingType state_);
	
	/** @brief Default destructor */
	~Button() = default;
	
	/**
	 * @brief Gets the button's sprite for rendering
	 * @return SFML sprite object for drawing the button
	 */
	sf::Sprite getSprite();
	
	/**
	 * @brief Checks if a mouse position intersects with the button
	 * @param mousePos Mouse position in screen coordinates
	 * @param boxPos Button's position in screen coordinates
	 * @return true if the button was clicked, false otherwise
	 */
	bool checkForClick(const sf::Vector2f mousePos, const sf::Vector2f boxPos);
};
} // namespace nn::visualizer

#endif // BUTTON
