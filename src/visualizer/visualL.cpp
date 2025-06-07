#include "visualL.hpp"
#include "fonts.hpp"

namespace nn {
namespace Visualizer {
visualL::visualL(const Layer &other, const std::shared_ptr<state> state_, const std::uint32_t width)
    : Layer(other),
      panel(state_),
      layerRender({width, NN_WIDTH}),
      WIDTH(width) {}

visualL::visualL(const int _size, const int _prev_size, const std::shared_ptr<state> state_, const std::uint32_t width)
    : Layer(_size, _prev_size, 0),
      panel(state_),
      layerRender({width, NN_WIDTH}),
      WIDTH(width) {}

void visualL::display() {
	layerRender.display();
}

void visualL::clear() {
	layerRender.clear(sf::Color::Transparent);
}

void visualL::do_render() {
	clear();
	drawNeurons();
	display();
}

sf::Sprite visualL::getSprite() {
	return sf::Sprite(layerRender.getTexture());
}

std::uint32_t visualL::calculateGap(const float size) {
	if (size <= 0)
		return 0;

	return (NN_HEIGHT - (size * NEURON_WIDTH)) / (size + 1);
}

float visualL::calculateDistance(const sf::Vector2f pos1, const sf::Vector2f pos2) {
	return sqrt(pow(pos1.x - pos2.x, 2) + pow(pos1.y - pos2.y, 2));
}

float visualL::calculateAngle(const sf::Vector2f pos1, const sf::Vector2f pos2) {
	return atan2(pos2.y - pos1.y, pos2.x - pos1.x) * 180.0 / M_PI;
}

void VParamLayer::drawWeights(const int neuron_i, const sf::Vector2f pos, const float prevGap) {
	const float FRACTION_ALONG_LINE = 0.8f;
	const float HORIZONTAL_SHIFT_PER_WEIGHT_TEXT = 4.0f;

	for (size_t neuronP = 0; neuronP < getPrevSize(); neuronP++) {
		float weightValue = Parameters.weights[neuron_i][neuronP];

		float xP = 0.f;
		float yP = prevGap + neuronP * (prevGap + NEURON_WIDTH);

		sf::Vector2f prevNeuronTopLeft(xP, yP);

		float lineLength = calculateDistance(prevNeuronTopLeft, pos);
		float angleDeg = calculateAngle(prevNeuronTopLeft, pos);
		float angleRad = angleDeg * M_PI / 180.0f;

		float line_thickness_arg = std::max(std::min(weightValue, 4.f), 0.2f);

		sf::RectangleShape line({lineLength, line_thickness_arg});
		line.setFillColor(sf::Color::Black);

		sf::Vector2f lineGraphicalOrigin(xP, yP + NEURON_RADIUS);
		line.setPosition(lineGraphicalOrigin);
		line.setRotation(sf::degrees(angleDeg));

		std::ostringstream ss;
		ss << std::fixed << std::setprecision(4) << weightValue;

		sf::Text text(Fonts::getFont());
		text.setCharacterSize(10);
		text.setString(ss.str());
		text.setFillColor(getColorFromTextT(getTextT(neuron_i, neuronP)));

		sf::FloatRect textBounds = text.getLocalBounds();
		text.setOrigin({textBounds.position.x + textBounds.size.x / 2.0f,
		                textBounds.position.y + textBounds.size.y / 2.0f});

		float text_anchor_local_x = FRACTION_ALONG_LINE * lineLength;
		float text_anchor_local_y = line_thickness_arg / 2.0f;

		float cosA = cosf(angleRad);
		float sinA = sinf(angleRad);

		float text_pos_x_transformed = lineGraphicalOrigin.x + text_anchor_local_x * cosA - text_anchor_local_y * sinA;
		float text_pos_y_transformed = lineGraphicalOrigin.y + text_anchor_local_x * sinA + text_anchor_local_y * cosA;

		float final_text_pos_x = text_pos_x_transformed - (neuronP * HORIZONTAL_SHIFT_PER_WEIGHT_TEXT);
		float final_text_pos_y = text_pos_y_transformed;

		text.setPosition({final_text_pos_x, final_text_pos_y});
		text.setRotation(sf::degrees(angleDeg));

		layerRender.draw(line);
		layerRender.draw(text);
	}
}

sf::Color VParamLayer::getColorFromTextT(const textT text_type) {
	if (text_type == textT::UP)
		return FONT_COLOR_UP;
	if (text_type == textT::DOWN)
		return FONT_COLOR_DOWN;
	return FONT_COLOR_NORMAL;
}

void visualL::drawNeurons() {
	float gap = calculateGap(getSize());
	float prevGap = calculateGap(getPrevSize());

	for (size_t neuron = 0; neuron < getSize(); neuron++) {
		renderNeuron(neuron, gap, prevGap);
	}
}

void VParamLayer::renderNeuron(const int index, const float gap, const float prevGap) {
	float x = WIDTH - NEURON_WIDTH;
	float y = gap + index * (gap + NEURON_WIDTH);

	drawWeights(index, {x, y}, prevGap);
	drawNeuron(dots.net[index], dots.out[index], {x, y});
}

void VEmptyLayer::renderNeuron(const int index, const float gap, const float) {
	float x = WIDTH - NEURON_WIDTH;
	float y = gap + index * (gap + NEURON_WIDTH);

	drawNeuron(dots.net[index], dots.out[index], {x, y});
}

textT VParamLayer::getTextT(const int layer_i, const int layer_p) {
	if (grad.weights[layer_i][layer_p] < 0)
		return textT::DOWN;
	if (grad.weights[layer_i][layer_p] > 0)
		return textT::UP;
	return textT::NORMAL;
}

textT VEmptyLayer::getTextT(const int, const int) {
	return textT::NORMAL;
}

textT visualL::getTextT(const int, const int) {
	return textT::NORMAL;
}

void visualL::drawNeuron(const double input, const double output, const sf::Vector2f pos) {
	sf::RectangleShape shape({NEURON_WIDTH, NEURON_WIDTH});
	shape.setFillColor(sf::Color::Blue);
	shape.setPosition(pos);

	std::ostringstream ss;
	ss << std::fixed << std::setprecision(4) << input << "\n"
	   << output;

	sf::Text text(Fonts::getFont());
	text.setCharacterSize(10);
	text.setString(ss.str());
	text.setFillColor(sf::Color::White);

	sf::FloatRect textBounds = text.getLocalBounds();
	text.setOrigin({textBounds.position.x + textBounds.size.x / 2.0f,
	                textBounds.position.y + textBounds.size.y / 2.0f});

	text.setPosition({pos.x + NEURON_RADIUS, pos.y + NEURON_RADIUS});

	layerRender.draw(shape);
	layerRender.draw(text);
}

void visualL::setDots(const std::vector<double> &out, const std::vector<double> &net) {
	set_update();
	dots.net = net;
	dots.out = out;
}

void visualL::set_weights(const LayerParameters &Param) {
	set_update();
	Parameters.set(Param);
}

void VParamLayer::updateGrad(const LayerParameters &new_grad) {
	set_update();
	grad.set(new_grad);
}
} // namespace Visualizer
} // namespace nn
