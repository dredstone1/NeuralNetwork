#include "graph.hpp"
#include "fonts.hpp"
#include <algorithm>

namespace Visualizer {
GraphUI::GraphUI(const std::shared_ptr<state> vstate_)
    : panel(vstate_),
      VRender({GRAPH_UI_WIDTH, GRAPH_HEIGHT}),
      Vgraph({GRAPH_WIDTH, GRAPH_HEIGHT}),
      currentData(0),
      graph_alpha(GRAPH_HEIGHT_ALPHA_DEFAULT) {
}

void GraphUI::render_numbers() {
	render_vertical_numbers();
}

void GraphUI::render_vertical_numbers() {
	sf::Text text(Fonts::getFont());
	text.setCharacterSize(10);
	text.setFillColor(GRAPH_VERTICAL_NUMBER_COLOR);

	for (int i = 0; i < VERTICAL_NUMBERS_COUNT; i++) {
		std::ostringstream number_str;
		number_str << std::fixed << std::setprecision(2) << (GRAPH_HEIGHT / graph_alpha) * (1.f - (i / (float)VERTICAL_NUMBERS_COUNT));

		text.setString(number_str.str());
		text.setPosition({5.f, i * ((float)GRAPH_HEIGHT / VERTICAL_NUMBERS_COUNT)});
		VRender.draw(text);
	}
}

sf::Sprite GraphUI::getSprite() {
	return sf::Sprite(VRender.getTexture());
}

void GraphUI::display() {
	Vgraph.display();
	sf::Sprite graph_sprite = sf::Sprite(Vgraph.getTexture());
	graph_sprite.setPosition({GRAPH_UI_WIDTH - GRAPH_WIDTH, 0});
	VRender.draw(graph_sprite);
	VRender.display();
}

void GraphUI::renderGraph() {
	render_numbers();
	for (size_t i = 0; i < resolution(); i++) {
		renderDot(i);
	}
}

void GraphUI::do_render() {
	clear();
	renderGraph();
	display();
}

void GraphUI::clear() {
	VRender.clear(GRAPH_BG);
	Vgraph.clear(GRAPH_BG);
}

int GraphUI::get_highest() {
	int max = 0;
	for (size_t i = 0; i < resolution(); i++) {
		if (data[i] > data[max]) {
			i = max;
		}
	}

	return max;
}

float GraphUI::get_hight(const int index) {
	return GRAPH_HEIGHT - std::max(1.f, static_cast<float>(data[index] * graph_alpha));
}

sf::Vector2f GraphUI::getPosition(const int index) {
	return sf::Vector2f(
	    index * data_gap_width(),
	    get_hight(index));
}

void GraphUI::renderDot(const int index) {
	sf::VertexArray line_(sf::PrimitiveType::Lines, 2);

	line_[0].position = getPosition(index);
	line_[1].position = getPosition(index + 1);

	if (GRAPH_HEIGHT < data[index] * graph_alpha) {
		graph_alpha = GRAPH_HEIGHT / data[index];
	}

	line_[0].color = GRAPH_LINE_COLOR;
	line_[1].color = GRAPH_LINE_COLOR;
	Vgraph.draw(line_);
}

std::uint32_t GraphUI::data_gap_width() {
	return std::max<std::uint32_t>(DATA_GAP_WIDTH, 1.f * GRAPH_WIDTH / resolution());
}

std::uint32_t GraphUI::resolution() {
	return std::min<std::uint32_t>(GRAPH_RESOLUTION, vstate->config.training_config.batch_count) - 1;
}

int GraphUI::data_gaps() {
	return vstate->config.training_config.batch_count / resolution();
}

int GraphUI::newDataPlace() {
	if (data_gaps() == 0) {
		return 0;
	}

	return std::floor((vstate->current_batch - 1) / data_gaps());
}

void GraphUI::add_data(const float new_data) {
	currentData = newDataPlace();
	data[currentData] += (new_data / data_gaps());
	set_update();
}
} // namespace Visualizer
