#include "visualL.hpp"
#include "fonts.hpp"
#include "model/LayerParameters.hpp"
#include "model/Layers/layer.hpp"
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/System/Vector2.hpp>
#include <SFML/Window/Keyboard.hpp>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <vector>

namespace Visualizer {
visualL::visualL(const Layer &other, const int size_a) : Layer(other), is_params(other.getPrevSize() != 0), WIDTH(calculateWIDTH(size_a, is_params)) {
	createLayerVisual();
}

visualL::visualL(const int _size, const int _prev_size, const int size_a) : Layer(_size, _prev_size, 0), is_params(_prev_size != 0), WIDTH(calculateWIDTH(size_a, is_params)) {
	createLayerVisual();
}

float visualL::calculateWIDTH(const int size_a, const bool is_params) {
	if (is_params && size_a <= 1) {
		return 2 * NEURON_RADIUS * 5;
	}
	return is_params ? (NN_WIDTH - NEURON_RADIUS * 2) / (size_a - 1.f) : 2 * NEURON_RADIUS;
}

void visualL::createLayerVisual() {
	layerRender.create(WIDTH, NN_HEIGHT);
}

void visualL::display() {
	layerRender.display();
}

void visualL::clear(const bool render) {
	layerRender.clear(getBGcolor(render));
}

sf::Color visualL::getBGcolor(const bool render) {
	if (render)
		return sf::Color::Yellow;

	return sf::Color::Magenta;
}

void visualL::renderLayer(const bool render) {
	clear(render);
	drawNeurons();
}

sf::Sprite visualL::getSprite() {
	display();
	return sf::Sprite(layerRender.getTexture());
}

float visualL::calculateGap(const float size) {
	if (size <= 0)
		return 0;

	return (NN_HEIGHT - (size * NEURON_RADIUS * 2)) / (size + 1);
}

float visualL::calculateDistance(const sf::Vector2f pos1, const sf::Vector2f pos2) {
	return sqrt(pow(pos1.x - pos2.x, 2) + pow(pos1.y - pos2.y, 2));
}

float visualL::calculateAngle(const sf::Vector2f pos1, const sf::Vector2f pos2) {
	return atan2(pos2.y - pos1.y, pos2.x - pos1.x) * 180.0 / M_PI;
}

void visualL::drawWeights(const int neuron_i, const sf::Vector2f pos, const float prevGap) {
	const float FRACTION_ALONG_LINE = 0.8f;
	const float HORIZONTAL_SHIFT_PER_WEIGHT_TEXT = 4.0f;

	for (int neuronP = 0; neuronP < getPrevSize(); neuronP++) {
		float weightValue = Parameters.weights[neuron_i][neuronP];

		float xP = 0.f;
		float yP = prevGap + neuronP * (prevGap + NEURON_RADIUS * 2);

		sf::Vector2f prevNeuronTopLeft(xP, yP);

		float lineLength = calculateDistance(prevNeuronTopLeft, pos);
		float angleDeg = calculateAngle(prevNeuronTopLeft, pos);
		float angleRad = angleDeg * M_PI / 180.0f;

		float line_thickness_arg = std::max(std::min(weightValue, 4.f), 0.1f);

		sf::RectangleShape line;
		line.setSize({lineLength, line_thickness_arg});
		line.setFillColor(sf::Color::Black);

		sf::Vector2f lineGraphicalOrigin(xP, yP + NEURON_RADIUS);
		line.setPosition(lineGraphicalOrigin);
		line.setRotation(angleDeg);

		layerRender.draw(line);

		std::ostringstream ss;
		ss << std::fixed << std::setprecision(4) << weightValue;

		sf::Text text;
		text.setFont(Fonts::getFont());
		text.setCharacterSize(10);
		text.setString(ss.str());
		text.setFillColor(getColorFromTextT(getTextT(neuron_i, neuronP)));

		sf::FloatRect textBounds = text.getLocalBounds();
		text.setOrigin(textBounds.left + textBounds.width / 2.0f,
		               textBounds.top + textBounds.height / 2.0f);

		float text_anchor_local_x = FRACTION_ALONG_LINE * lineLength;
		float text_anchor_local_y = line_thickness_arg / 2.0f;

		float cosA = cosf(angleRad);
		float sinA = sinf(angleRad);

		float text_pos_x_transformed = lineGraphicalOrigin.x + text_anchor_local_x * cosA - text_anchor_local_y * sinA;
		float text_pos_y_transformed = lineGraphicalOrigin.y + text_anchor_local_x * sinA + text_anchor_local_y * cosA;

		float final_text_pos_x = text_pos_x_transformed - (neuronP * HORIZONTAL_SHIFT_PER_WEIGHT_TEXT);
		float final_text_pos_y = text_pos_y_transformed;

		text.setPosition(final_text_pos_x, final_text_pos_y);
		text.setRotation(angleDeg);

		layerRender.draw(text);
	}
}

sf::Color visualL::getColorFromTextT(const textT text_type) {
	if (text_type == textT::UP)
		return sf::Color::Red;
	if (text_type == textT::DOWN)
		return sf::Color::Blue;
	return sf::Color(50, 50, 50);
}

void visualL::drawNeurons() {
	float gap = calculateGap(getSize());
	float prevGap = calculateGap(getPrevSize());

	for (int neuron = 0; neuron < getSize(); neuron++) {
		float x = WIDTH - NEURON_RADIUS * 2;
		float y = gap + neuron * (gap + NEURON_RADIUS * 2);

		if (is_params)
			drawWeights(neuron, {x, y}, prevGap);

		drawNeuron(dots.net[neuron], dots.out[neuron], {x, y});
	}
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
	sf::RectangleShape shape({NEURON_RADIUS * 2, NEURON_RADIUS * 2});
	shape.setFillColor(sf::Color::Blue);
	shape.setPosition(pos);

	std::ostringstream ss;
	ss << std::fixed << std::setprecision(4) << input << "\n"
	   << output;

	sf::Text text;
	text.setFont(Fonts::getFont());
	text.setCharacterSize(10);
	text.setString(ss.str());
	text.setFillColor(sf::Color::White);

	sf::FloatRect textBounds = text.getLocalBounds();
	text.setOrigin(textBounds.left + textBounds.width / 2.0f,
	               textBounds.top + textBounds.height / 2.0f);

	text.setPosition(pos.x + NEURON_RADIUS, pos.y + NEURON_RADIUS);

	layerRender.draw(shape);
	layerRender.draw(text);
}

void visualL::setDots(const std::vector<double> out, const std::vector<double> net) {
	dots.net = net;
	dots.out = out;
}

void VParamLayer::updateGrad(const LayerParameters &new_grad) {
	grad.reset();
	grad.add(new_grad);
}
} // namespace Visualizer
