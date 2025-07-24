#include "graph.hpp"
#include "fonts.hpp"

namespace nn::visualizer {
Graph::Graph(const int batchCount, std::uint32_t res, float alpha)
    : graphAlpha(alpha),
      resolution(std::min(GRAPH_RESOLUTION, res) - 1),
      dataGaps(batchCount / resolution) {
}

float Graph::getHeight(const float value) const {
	return GRAPH_HEIGHT - std::max(1.0, value * graphAlpha);
}

sf::Vector2f Graph::getPosition(int index) const {
	return {index * dataGapWidth(), getHeight(getValue(index))};
}

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
		renderDot(i, target, position, color);
	}
}

void Graph::addData(global::ValueType new_data, int index) {

	if (dataGaps == 0)
		return;

	int place = std::floor((index - 1) / dataGaps);
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
      graphLost(vstate->config.trainingConfig.getBatchCount(), GRAPH_RESOLUTION, GRAPH_HEIGHT_ALPHA_DEFAULT),
      graphEvaluate(vstate->config.trainingConfig.getBatchCount(), GRAPH_RESOLUTION, GRAPH_HEIGHT_ALPHA_DEFAULT / 100) {
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

void GraphUIPanel::addData(
    const global::ValueType newDataEvaluate,
    const global::ValueType newDataLost,
    int index) {
	graphLost.addData(newDataLost, index);
	graphEvaluate.addData(newDataEvaluate, index);
	setUpdate();
}
} // namespace nn::visualizer
