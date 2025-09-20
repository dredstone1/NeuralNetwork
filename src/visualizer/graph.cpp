/**
 * @file graph.cpp
 * @brief Implementation of graph visualization components for training metrics
 * 
 * This file implements real-time graph visualization for tracking neural network
 * training progress. It provides functionality for plotting loss curves and
 * evaluation metrics over time, with automatic scaling and data management.
 */

#include "graph.hpp"
#include "fonts.hpp"

namespace nn::visualizer {

/**
 * @brief Constructs a Graph for displaying training metrics over time
 * 
 * Creates a graph visualization component that can display data points
 * over the course of training. The graph automatically scales and manages
 * data points within a fixed resolution window.
 * 
 * @param batchCount Total number of training batches (for data gap calculation)
 * @param res Resolution of the graph (maximum number of data points displayed)
 * @param alpha Scaling factor for graph height visualization
 */
Graph::Graph(const int batchCount, std::uint32_t res, float alpha)
    : graphAlpha(alpha),
      resolution(std::min(GRAPH_RESOLUTION, res) - 1),
      dataGaps(batchCount / resolution) {
}

/**
 * @brief Calculates the vertical position for a given value on the graph
 * 
 * Converts a data value to screen coordinates, applying scaling and ensuring
 * the point fits within the graph's height bounds.
 * 
 * @param value The data value to convert to screen coordinates
 * @return Y-coordinate position for rendering (inverted, as screen Y increases downward)
 */
float Graph::getHeight(const float value) const {
	return GRAPH_HEIGHT - std::max(1.0, value * graphAlpha);
}

/**
 * @brief Calculates the 2D position for a data point at the given index
 * 
 * Converts a data index to screen coordinates for rendering, combining
 * horizontal positioning based on time/index and vertical positioning
 * based on the data value.
 * 
 * @param index Index of the data point in the graph's data array
 * @return 2D position vector for rendering this data point
 */
sf::Vector2f Graph::getPosition(int index) const {
	return {index * dataGapWidth(), getHeight(getValue(index))};
}

/**
 * @brief Calculates the horizontal spacing between data points
 * 
 * @return Width in pixels between consecutive data points on the graph
 */
float Graph::dataGapWidth() const {
	return GRAPH_WIDTH / static_cast<float>(resolution);
}

global::ValueType Graph::getValue(const int index) const {
	if (Index == index) {
		return data[index] / IndexCount;
	}

	return data[index] / dataGaps;
}

void Graph::renderDot(
    const int index,
    sf::RenderTarget &target,
    const sf::Vector2f &position,
    const sf::Color &color) {
	if (GRAPH_HEIGHT < getValue(index) * graphAlpha) {
		graphAlpha = GRAPH_HEIGHT / getValue(index);
	}

	sf::VertexArray line(sf::PrimitiveType::Lines, 2);
	line[0].position = getPosition(index) + position;
	line[1].position = getPosition(index + 1) + position;
	line[0].color = color;
	line[1].color = color;

	target.draw(line);
}

void Graph::drawTo(sf::RenderTarget &target, sf::Vector2f position, const sf::Color &color) {
	for (int i = 0; i < static_cast<int>(resolution); ++i) {
		if (data[i + 1] == 0) {
			break;
		}

		renderDot(i, target, position, color);
	}
}

void Graph::addData(global::ValueType new_data, int index) {
	if (dataGaps == 0)
		return;

	int place = std::floor(index / dataGaps);

	if (place < 0 || place >= static_cast<int>(resolution))
		return;

	if (Index != place) {
		IndexCount = 0;
	}

	Index = place;

	data[place] += new_data;
	IndexCount++;
}

void Graph::setAlpha(float alpha) {
	graphAlpha = alpha;
}

float Graph::getAlpha() const {
	return graphAlpha;
}

GraphUIPanel::GraphUIPanel(std::shared_ptr<StateManager> vstate_)
    : Panel(vstate_),
      VRender({GRAPH_UI_WIDTH, GRAPH_HEIGHT}),
      graphLost(
          vstate->config.trainingConfig.getBatchCount(),
          GRAPH_RESOLUTION,
          GRAPH_HEIGHT_ALPHA_DEFAULT),
      graphEvaluate(
          vstate->config.trainingConfig.getBatchCount() / vstate_->config.trainingConfig.getAutoEvaluating().evaluateEvery,
          GRAPH_RESOLUTION,
          GRAPH_HEIGHT_ALPHA_DEFAULT / 100) {
}

void GraphUIPanel::clear() {
	VRender.clear(GRAPH_BG);
}

void GraphUIPanel::renderVerticalNumbers() {
	sf::Text textE(Fonts::getFont());
	sf::Text textL(Fonts::getFont());
	textE.setCharacterSize(10);
	textL.setCharacterSize(10);
	textE.setFillColor(GRAPH_LINE_COLOR_EVALUATE);
	textL.setFillColor(GRAPH_LINE_COLOR_LOST);

	for (int i = 0; i < VERTICAL_NUMBERS_COUNT; ++i) {
		float y = i * (GRAPH_HEIGHT / (float)VERTICAL_NUMBERS_COUNT) + 15 + 4;

		float valueE = getValueFromHeightE(y);
		float valueL = getValueFromHeightL(y);

		std::ostringstream ssE, ssL;
		ssE << std::fixed << std::setprecision(2) << valueE;
		ssL << std::fixed << std::setprecision(2) << valueL;

		textE.setString(ssE.str());
		textL.setString(ssL.str());

		sf::FloatRect boundsE = textE.getLocalBounds();

		textE.setPosition({5.f, y});
		textL.setPosition({5.f + boundsE.size.x + 5.f, y});

		textE.setOrigin({0, textE.getLocalBounds().getCenter().y});
		textL.setOrigin({0, textL.getLocalBounds().getCenter().y});

		VRender.draw(textE);
		VRender.draw(textL);

		renderHorizontalLine(valueL);
	}
}

void GraphUIPanel::renderHorizontalLine(float value) {
	float pos_y = GRAPH_HEIGHT - value * graphLost.getAlpha();
	std::array line{
	    sf::Vertex{sf::Vector2f(GRAPH_UI_WIDTH - GRAPH_WIDTH, pos_y), GRAPH_HORIZONTAL_LINE_COLOR},
	    sf::Vertex{sf::Vector2f(GRAPH_UI_WIDTH, pos_y), GRAPH_HORIZONTAL_LINE_COLOR}};

	VRender.draw(line.data(), line.size(), sf::PrimitiveType::Lines);
}

float GraphUIPanel::getValueFromHeightL(float height) {
	return (GRAPH_HEIGHT - height) / graphLost.getAlpha();
}
float GraphUIPanel::getValueFromHeightE(float height) {
	return (GRAPH_HEIGHT - height) / graphEvaluate.getAlpha();
}

void GraphUIPanel::display() {
	VRender.display();
}

void GraphUIPanel::doRender() {
	clear();

	graphLost.drawTo(VRender, {GRAPH_UI_WIDTH - GRAPH_WIDTH, 0}, GRAPH_LINE_COLOR_LOST);

	graphEvaluate.drawTo(VRender, {GRAPH_UI_WIDTH - GRAPH_WIDTH, 0}, GRAPH_LINE_COLOR_EVALUATE);

	renderVerticalNumbers();
	display();
}

sf::Sprite GraphUIPanel::getSprite() {
	return sf::Sprite(VRender.getTexture());
}

void GraphUIPanel::addEvaluateData(const global::ValueType newDataEvaluate, int index) {
	graphEvaluate.addData(newDataEvaluate, index);
	setUpdate();
}

void GraphUIPanel::addLostData(const global::ValueType newDataLost, int index) {
	graphLost.addData(newDataLost, index);
	setUpdate();
}

void Graph::reset() {
	Index = 0;
	IndexCount = 0;
    data = {0};
}

void GraphUIPanel::reset() {
    graphLost.reset();
    graphEvaluate.reset();
}
} // namespace nn::visualizer
