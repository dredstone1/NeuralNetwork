/**
 * @file panel.hpp
 * @brief Base class for UI panel components in the neural network visualizer
 * 
 * This file defines the Panel base class that provides common functionality
 * for all UI components in the visualization system. It handles update
 * management, state observation, and provides a consistent interface for
 * rendering visual components.
 */

#ifndef PANEL
#define PANEL

#include "state.hpp"
#include <SFML/Graphics/Color.hpp>

namespace nn::visualizer {

/// Standard background color for all panel components
constexpr sf::Color PANELS_BG(255, 255, 255);

/**
 * @class Panel
 * @brief Abstract base class for all UI panel components
 * 
 * The Panel class provides the foundational structure for all visual components
 * in the neural network visualizer. It manages update states, provides access
 * to the shared state manager, and defines the interface that all UI components
 * must implement.
 * 
 * Key features:
 * - Update management to avoid unnecessary re-rendering
 * - Integration with the state management system
 * - Virtual interface for custom rendering implementations
 * - Observer pattern support for state changes
 * 
 * Derived classes must implement the doRender() method to define their
 * specific rendering behavior.
 */
class Panel {
  private:
	/** @brief Pure virtual method for rendering panel content - must be implemented by derived classes */
	virtual void doRender() = 0;
	
	bool need_update{true};  ///< Flag indicating whether the panel needs to be re-rendered
	
	/** @brief Virtual method for observing state changes - can be overridden by derived classes */
	virtual void observe() {}

  protected:
	std::shared_ptr<StateManager> vstate; ///< Shared state manager for this visualizer session

  public:
	/**
	 * @brief Constructs a panel with access to the state manager
	 * @param vstate_ Shared pointer to the state manager
	 */
	Panel(const std::shared_ptr<StateManager> vstate_)
	    : vstate(vstate_) {}
	
	/** @brief Virtual destructor for proper cleanup of derived classes */
	virtual ~Panel() = default;

	/**
	 * @brief Renders the panel if an update is needed
	 * 
	 * Checks if the panel needs updating and calls the doRender() method
	 * if necessary. Also calls observe() to check for state changes.
	 * 
	 * @return Status code indicating render result
	 */
	int render();
	
	/**
	 * @brief Checks if the panel needs to be updated
	 * @return true if the panel needs re-rendering, false otherwise
	 */
	virtual bool updateStatus() const { return need_update; }
	
	/**
	 * @brief Marks the panel as needing an update
	 * @param wait Optional parameter for update timing control
	 */
	void setUpdate(const bool wait = false);
};
} // namespace nn::visualizer

#endif // PANEL
