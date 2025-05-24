#include "visualL.hpp"
#include "fonts.hpp"
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
visualL::visualL(Layer const &other, const int size_a) : Layer(other.getSize(), other.getPrevSize(), false), is_params(other.getPrevSize() != 0), WIDTH(calculateWIDTH(size_a, is_params)) {
	createLayerVisual();
}

void visualL::forward(const vector<double> &metrix) {
    if (metrix[0] == 0) {

    }
}

visualL::visualL(int _size, int _prev_size, const int size_a) : Layer(_size, _prev_size), is_params(_prev_size != 0), WIDTH(calculateWIDTH(size_a, is_params)) {
	createLayerVisual();
}

float visualL::calculateWIDTH(const int size_a, const bool is_params) {
	return ((is_params) ? (NN_WIDTH - NEURON_RADIUS * 2) / (size_a - 1.f) : 2 * NEURON_RADIUS);
}

void visualL::createLayerVisual() {
	layerRender.create(WIDTH, NN_HEIGHT);
}

void visualL::display() {
	layerRender.display();
}

void visualL::clear() {
	layerRender.clear(sf::Color::Transparent);
}

void visualL::renderLayer() {
	clear();
	drawNeurons();
}

sf::Sprite visualL::getSprite() {
	display();
	return sf::Sprite(layerRender.getTexture());
}

float visualL::calculateGap(const float size) {
	return (NN_HEIGHT - (size * NEURON_RADIUS * 2)) / (size + 1);
}

float visualL::calculateDistance(sf::Vector2f pos1, sf::Vector2f pos2) {
	return sqrt(pow(pos1.x - pos2.x, 2) + pow(pos1.y - pos2.y, 2));
}

float visualL::calculateAngle(sf::Vector2f pos1, sf::Vector2f pos2) {
	double angleRadians = atan2(pos2.y - pos1.y, pos2.x - pos1.x);
	double angleDegrees = angleRadians * 180.0 / M_PI;
	return angleDegrees;
}

void visualL::drawWeights(int neuron_i, sf::Vector2f pos, float prevGap) {
	for (int neuronP = 0; neuronP < getPrevSize(); neuronP++) {
		if (Parameters->weights[neuron_i][neuronP] < 0) {
			continue;
		}

		float xP = 0.f;
		float yP = prevGap + neuronP * (prevGap + NEURON_RADIUS * 2);

		float width = calculateDistance(pos, {xP, yP});
		float angle = calculateAngle({xP, yP}, pos);

		sf::RectangleShape line({width, 1});
		line.setFillColor(sf::Color::Black);
		line.setPosition(xP, yP + NEURON_RADIUS);
		line.setRotation(angle);

		float Size = (float)Parameters->weights[neuron_i][neuronP];
		line.setSize({line.getSize().x, Size});
		layerRender.draw(line);
	}
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

void visualL::drawNeuron(const double input, const double output, sf::Vector2f pos) {
	sf::RectangleShape shape({NEURON_RADIUS * 2, NEURON_RADIUS * 2});
	shape.setFillColor(sf::Color::Blue);
	shape.setPosition(pos);

	ostringstream ss;
	ss << fixed << setprecision(4) << input << "\n"
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

void visualL::setDots(vector<double> out, vector<double> net) {
	dots.net = net;
	dots.out = out;
}
} // namespace Visualizer
