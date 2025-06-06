#include "graph.hpp"
#include "Globals.hpp"
#include "fonts.hpp"
#include <SFML/Graphics/Color.hpp>
#include <algorithm>
#include <iostream>
#include <ostream>

namespace nn {
namespace Visualizer {
GraphUI::GraphUI(const std::shared_ptr<state> vstate_)
    : panel(vstate_),
      VRender({GRAPH_UI_WIDTH, GRAPH_HEIGHT}),
      Vgraph({GRAPH_WIDTH, GRAPH_HEIGHT}),
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

		text.setOrigin({0, 0});
		text.setPosition({5.f, i * ((float)GRAPH_HEIGHT / VERTICAL_NUMBERS_COUNT) + 15 + 4});
		float value = get_value_from_height(text.getPosition().y - text.getOrigin().y);
		number_str << std::fixed << std::setprecision(2) << value;

		text.setString(number_str.str());
		text.setOrigin({0, text.getLocalBounds().getCenter().y});

		VRender.draw(text);
		render_horizontal_line(value);
	}
}

void GraphUI::render_horizontal_line(const float value) {
	float pos_y = get_height(value);
	std::array line{
	    sf::Vertex{sf::Vector2f(0, pos_y)},
	    sf::Vertex{sf::Vector2f(GRAPH_WIDTH, pos_y)}};

	line[0].color = sf::Color::Blue;
	line[1].color = sf::Color::Blue;

	Vgraph.draw(line.data(), line.size(), sf::PrimitiveType::Lines);
}

float GraphUI::get_value_from_height(const float height) {
	return (GRAPH_HEIGHT - height) / graph_alpha;
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

float GraphUI::get_height(const float value) {
	return GRAPH_HEIGHT - std::max(1.0, value * graph_alpha);
}

float GraphUI::get_height(const int index) {
	return get_height((float)data[index]);
}

sf::Vector2f GraphUI::getPosition(const int index) {
	return sf::Vector2f(
	    index * data_gap_width(),
	    get_height(index));
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

float GraphUI::data_gap_width() {
	return GRAPH_WIDTH / (float)resolution();
}

std::uint32_t GraphUI::resolution() {
	return std::min(GRAPH_RESOLUTION, (std::uint32_t)vstate->config.training_config.batch_count) - 1;
}

int GraphUI::data_gaps() {
	return vstate->config.training_config.batch_count / resolution();
}

int GraphUI::newDataPlace(const int index) {
	if (data_gaps() == 0) {
		return 0;
	}

	return std::floor((index - 1) / data_gaps());
}

void GraphUI::add_data(const Global::ValueType new_data, const int index) {
	data[newDataPlace(index)] += (new_data / data_gaps());
	set_update();
}
} // namespace Visualizer
} // namespace nn
