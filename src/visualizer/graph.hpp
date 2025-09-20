/**
 * @file graph.hpp
 * @brief Header file for graph visualization components in the neural network visualizer
 *
 * This file defines classes and constants for creating real-time graphs that display
 * training metrics such as loss curves and evaluation accuracy over time. The graph
 * system provides automatic scaling, data management, and smooth visualization of
 * training progress.
 */

#ifndef GRAPH_CORE_HPP
#define GRAPH_CORE_HPP

#include "panel.hpp"
#include <SFML/Graphics.hpp>

namespace nn::visualizer {

// Graph dimension and display constants
constexpr std::uint32_t GRAPH_WIDTH = 440;      ///< Graph plotting area width in pixels
constexpr std::uint32_t GRAPH_HEIGHT = 315;     ///< Graph plotting area height in pixels
constexpr std::uint32_t GRAPH_RESOLUTION = 100; ///< Maximum number of data points displayed
constexpr int VERTICAL_NUMBERS_COUNT = 10;      ///< Number of vertical axis labels

constexpr std::uint32_t GRAPH_UI_WIDTH = 500; ///< Total graph UI panel width

// Graph styling constants
constexpr int GRAPH_TEXT_FONT = 30;                                      ///< Font size for graph labels
constexpr std::uint32_t DATA_GAP_WIDTH = GRAPH_WIDTH / GRAPH_RESOLUTION; ///< Spacing between data points
constexpr float GRAPH_HEIGHT_ALPHA_DEFAULT = GRAPH_HEIGHT;               ///< Default scaling factor

// Graph color scheme
constexpr sf::Color GRAPH_LINE_COLOR_LOST(0, 0, 0);         ///< Loss curve color (black)
constexpr sf::Color GRAPH_LINE_COLOR_EVALUATE(255, 0, 255); ///< Evaluation curve color (magenta)
constexpr sf::Color GRAPH_HORIZONTAL_LINE_COLOR(0, 0, 255); ///< Grid line color (blue)
constexpr sf::Color GRAPH_BG = PANELS_BG;                   ///< Graph background color

/**
 * @class Graph
 * @brief Core graph component for plotting training metrics over time
 *
 * The Graph class manages the visualization of time-series data during neural
 * network training. It handles data point storage, automatic scaling, and
 * rendering of line graphs with appropriate colors and positioning.
 *
 * Key features:
 * - Fixed-resolution data storage with automatic aggregation
 * - Dynamic vertical scaling based on data range
 * - Smooth line rendering between data points
 * - Configurable colors and styling
 */
class Graph {
  private:
	std::array<global::ValueType, GRAPH_RESOLUTION> data{0}; ///< Data storage array
	double graphAlpha;                                       ///< Vertical scaling factor for data visualization
	std::uint32_t resolution;                                ///< Active resolution (may be less than max)

	int IndexCount{0};  ///< Count of data points at current index
	int Index{0};       ///< Current data insertion index
	const int dataGaps; ///< Number of raw data points per graph point

	/** @brief Gets the aggregated value at a specific graph index */
	global::ValueType getValue(const int index) const;

	/** @brief Converts a data value to screen Y-coordinate */
	float getHeight(const float value) const;

	/** @brief Gets the 2D screen position for a data point */
	sf::Vector2f getPosition(const int index) const;

	/**
	 * @brief Renders a line segment between two consecutive data points
	 * @param index Starting index for the line segment
	 * @param target Render target to draw on
	 * @param position Offset position for the graph
	 * @param color Color for the line segment
	 */
	void renderDot(
	    const int index,
	    sf::RenderTarget &target,
	    const sf::Vector2f &position,
	    const sf::Color &color);

	/** @brief Calculates horizontal spacing between data points */
	float dataGapWidth() const;

  public:
	/**
	 * @brief Constructs a graph with specified parameters
	 * @param batchCount Total number of training batches
	 * @param resolution Maximum resolution (data points) for the graph
	 * @param alpha Vertical scaling factor for value display
	 */
	Graph(const int batchCount, std::uint32_t resolution = GRAPH_RESOLUTION, float alpha = GRAPH_HEIGHT);

	/** @brief Default destructor */
	~Graph() = default;

	/**
	 * @brief Draws the complete graph to a render target
	 * @param rarget Render target to draw the graph on
	 * @param position Offset position for graph placement
	 * @param color Color for the graph lines
	 */
	void drawTo(sf::RenderTarget &rarget, const sf::Vector2f position = {0.f, 0.f}, const sf::Color &color = sf::Color::Black);

	/**
	 * @brief Adds a new data point to the graph
	 * @param new_data The data value to add
	 * @param index The batch/time index for this data point
	 */
	void addData(const global::ValueType new_data, const int index);
	void setAlpha(const float alpha);
	float getAlpha() const;

	void reset();
};

class GraphUIPanel : public Panel {
  private:
	sf::RenderTexture VRender;
	Graph graphLost;
	Graph graphEvaluate;

	void renderVerticalNumbers();
	void renderHorizontalLine(float value);
	float getValueFromHeightE(float height);
	float getValueFromHeightL(float height);
	void display();
	void clear();
	void doRender() override;

  public:
	GraphUIPanel(std::shared_ptr<StateManager> vstate_);
	~GraphUIPanel() = default;

	sf::Sprite getSprite();

	void addLostData(const global::ValueType newDataLost, int index);
	void addEvaluateData(const global::ValueType newDataEvaluate, int index);

	void reset();
};

} // namespace nn::visualizer

#endif // GRAPH_CORE_HPP
