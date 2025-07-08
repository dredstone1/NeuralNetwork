#include "graph.hpp"
#include "fonts.hpp"

namespace nn::visualizer {
GraphUIPanel::GraphUIPanel(const std::shared_ptr<StateManager> vstate_)
    : Panel(vstate_),
      VRender({GRAPH_UI_WIDTH, GRAPH_HEIGHT}),
      Vgraph({GRAPH_WIDTH, GRAPH_HEIGHT}),
      graphAlpha(GRAPH_HEIGHT_ALPHA_DEFAULT) {
}

void GraphUIPanel::renderVerticalNumbers() {
	sf::Text text(Fonts::getFont());
	text.setCharacterSize(10);
	text.setFillColor(GRAPH_VERTICAL_NUMBER_COLOR);

	for (int i = 0; i < VERTICAL_NUMBERS_COUNT; ++i) {
		std::ostringstream number_str;

		text.setOrigin({0, 0});
		text.setPosition({5.f, i * ((float)GRAPH_HEIGHT / VERTICAL_NUMBERS_COUNT) + 15 + 4});
		float value = getValueFromHeight(text.getPosition().y - text.getOrigin().y);
		number_str << std::fixed << std::setprecision(2) << value;

		text.setString(number_str.str());
		text.setOrigin({0, text.getLocalBounds().getCenter().y});

		VRender.draw(text);
		renderHorizontalLine(value);
	}
}

void GraphUIPanel::renderHorizontalLine(const float value) {
	float pos_y = getHeight(value);
	std::array line{
	    sf::Vertex{sf::Vector2f(0, pos_y), GRAPH_HORIZONTAL_LINE_COLOR},
	    sf::Vertex{sf::Vector2f(GRAPH_WIDTH, pos_y), GRAPH_HORIZONTAL_LINE_COLOR}};

	Vgraph.draw(line.data(), line.size(), sf::PrimitiveType::Lines);
}

float GraphUIPanel::getValueFromHeight(const float height) {
	return (GRAPH_HEIGHT - height) / graphAlpha;
}

sf::Sprite GraphUIPanel::getSprite() {
	return sf::Sprite(VRender.getTexture());
}

void GraphUIPanel::display() {
	Vgraph.display();

	sf::Sprite graph_sprite = sf::Sprite(Vgraph.getTexture());
	graph_sprite.setPosition({GRAPH_UI_WIDTH - GRAPH_WIDTH, 0});
	VRender.draw(graph_sprite);
	VRender.display();
}

void GraphUIPanel::renderGraph() {
	renderVerticalNumbers();

	for (size_t i = 0; i < resolution(); ++i) {
		renderDot(i);
	}
}

void GraphUIPanel::doRender() {
	clear();
	renderGraph();
	display();
}

void GraphUIPanel::clear() {
	VRender.clear(GRAPH_BG);
	Vgraph.clear(GRAPH_BG);
}

float GraphUIPanel::getHeight(const float value) {
	return GRAPH_HEIGHT - std::max(1.0, value * graphAlpha);
}

float GraphUIPanel::getHeight(const int index) {
	return getHeight((float)data[index]);
}

sf::Vector2f GraphUIPanel::getPosition(const int index) {
	return sf::Vector2f(index * dataGapWidth(), getHeight(index));
}

void GraphUIPanel::renderDot(const int index) {
	sf::VertexArray line_(sf::PrimitiveType::Lines, 2);

	line_[0].position = getPosition(index);
	line_[1].position = getPosition(index + 1);

	if (GRAPH_HEIGHT < data[index] * graphAlpha) {
		graphAlpha = GRAPH_HEIGHT / data[index];
	}

	line_[0].color = GRAPH_LINE_COLOR;
	line_[1].color = GRAPH_LINE_COLOR;

	Vgraph.draw(line_);
}

float GraphUIPanel::dataGapWidth() {
	return GRAPH_WIDTH / (float)resolution();
}

std::uint32_t GraphUIPanel::resolution() {
	return std::min(GRAPH_RESOLUTION, (std::uint32_t)vstate->config.trainingConfig.batch_count) - 1;
}

int GraphUIPanel::dataGaps() {
	return vstate->config.trainingConfig.batch_count / resolution();
}

int GraphUIPanel::newDataPlace(const int index) {
	if (dataGaps() == 0) {
		return 0;
	}

	return std::floor((index - 1) / dataGaps());
}

void GraphUIPanel::addData(const global::ValueType new_data, const int index) {
	data[newDataPlace(index)] += new_data / dataGaps();
	setUpdate();
}

} // namespace nn::visualizer
