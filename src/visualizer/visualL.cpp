#include "visualL.hpp"
#include "fonts.hpp"
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/Rect.hpp>
#include <SFML/System/Angle.hpp>
#include <SFML/System/Vector2.hpp>
#include <SFML/Window/Keyboard.hpp>

namespace nn::visualizer {
visualLayer::visualLayer(const int _size, const int _prev_size, const std::shared_ptr<StateManager> state_, const std::uint32_t width)
    : ILayer(_size, _prev_size, 0),
      Panel(state_),
      WIDTH(width),
      layerRender({width, NN_WIDTH}),
      cacheNeurons(_size) {
	doCacheNeurons();
}

void visualLayer::doCacheNeurons() {
	float scale = getScaleFactor(getSize());
	float neuron_width_scaled = NEURON_WIDTH * scale;

	float gap = calculateGap(getSize(), neuron_width_scaled);
	float x = WIDTH - neuron_width_scaled;

	for (size_t neuron = 0; neuron < getSize(); neuron++) {
		float y = neuron * (gap + neuron_width_scaled);
		cacheNeurons[neuron] = sf::FloatRect({x, y}, {neuron_width_scaled, neuron_width_scaled});
	}
}

void visualLayer::display() {
	layerRender.display();
}

void visualLayer::clear() {
	layerRender.clear(sf::Color::Transparent);
}

void visualLayer::doRender() {
	clear();
	drawNeurons();
	display();
}

sf::Sprite visualLayer::getSprite() {
	return sf::Sprite(layerRender.getTexture());
}

float visualLayer::calculateDistance(const sf::Vector2f pos1, const sf::Vector2f pos2) {
	return sqrt(pow(pos1.x - pos2.x, 2) + pow(pos1.y - pos2.y, 2));
}

sf::Angle visualLayer::calculateAngle(const sf::Vector2f pos1, const sf::Vector2f pos2) {
	return sf::radians(atan2(pos2.y - pos1.y, pos2.x - pos1.x));
}

float visualLayer::getScaleFactor(std::size_t neuron_count) {
	float maxNeuronSpace = NN_HEIGHT - (neuron_count)*MIN_GAP;

	float neuronWidth = maxNeuronSpace / std::max<float>(neuron_count, 1);
	neuronWidth = std::clamp(neuronWidth, MIN_NEURON_WIDTH, MAX_NEURON_WIDTH);

	return neuronWidth / MAX_NEURON_WIDTH;
}

void visualParamLayer::drawWeights(const int neuron_i) {
	for (size_t neuronP = 0; neuronP < getPrevSize(); neuronP++) {
		sf::VertexArray line_(sf::PrimitiveType::LineStrip, 3);

		line_[0].position = {0, cachePrevNeurons[neuronP].position.y};
		line_[1].position = {cacheNeurons[neuron_i].position.x / 2, (cacheNeurons[neuron_i].position.y + cachePrevNeurons[neuronP].position.y) / 2};
		line_[2].position = cacheNeurons[neuron_i].position;

		line_[0].color = LINE_COLOR;
		line_[0].color.a = parameters.weights[neuron_i][neuronP] * 50;
		line_[1].color = line_[0].color;
		line_[2].color = getColorFromTextT(getTextT(neuron_i, neuronP));
		layerRender.draw(line_);
	}
}

void visualParamLayer::doCacheWeights() {
	float scale = getScaleFactor(getPrevSize());
	float neuron_width_scaled = NEURON_WIDTH * scale;

	float gap = calculateGap(getPrevSize(), neuron_width_scaled);
	float x = WIDTH - neuron_width_scaled;

	for (size_t neuron = 0; neuron < getPrevSize(); neuron++) {
		float y = neuron * (gap + neuron_width_scaled);
		cachePrevNeurons[neuron] = sf::FloatRect({x, y}, {neuron_width_scaled, neuron_width_scaled});
	}
}

sf::Color visualParamLayer::getColorFromTextT(const textType text_type) {
	return color_lookup[static_cast<size_t>(text_type)];
}

float visualLayer::calculateGap(const int size, const float scale) {
	return (NN_HEIGHT - (size * scale)) / (size - 1);
}

void visualLayer::drawNeurons() {

	for (size_t neuron = 0; neuron < getSize(); neuron++) {
		renderNeuron(neuron);
	}
}

void visualParamLayer::renderNeuron(const int index) {
	drawWeights(index);
	drawNeuron(cacheNeurons[index], dots.net[index], dots.out[index]);
}

void visualEmptyLayer::renderNeuron(const int index) {
	drawNeuron(cacheNeurons[index], dots.net[index], dots.out[index]);
}

textType visualParamLayer::getTextT(const int layer_i, const int layer_p) {
	if (grad.weights[layer_i][layer_p] < 0)
		return textType::DOWN;
	if (grad.weights[layer_i][layer_p] > 0)
		return textType::UP;
	return textType::NORMAL;
}

textType visualEmptyLayer::getTextT(const int, const int) {
	return textType::NORMAL;
}

textType visualLayer::getTextT(const int, const int) {
	return textType::NORMAL;
}

sf::Color visualLayer::getNeuronColor(const global::ValueType value) {
	sf::Color newColor = NEURON_BG_COLOR;
	newColor.b *= value;
	return newColor;
}

void visualLayer::drawNeuron(const sf::FloatRect &rect, const double input, const double output) {
	sf::RectangleShape shape(rect.size);
	shape.setFillColor(getNeuronColor(output));
	shape.setPosition(rect.position);

	std::ostringstream ss;
	ss << std::fixed << std::setprecision(4) << input << std::endl
	   << output;

	sf::Text text(Fonts::getFont());
	text.setCharacterSize(10 * rect.size.y / NEURON_WIDTH);
	text.setString(ss.str());
	text.setFillColor(NEURON_TEXT_COLOR);

	sf::FloatRect textBounds = text.getLocalBounds();
	text.setOrigin({textBounds.position.x + textBounds.size.x / 2.0f,
	                textBounds.position.y + textBounds.size.y / 2.0f});

	text.setPosition({rect.position.x + rect.size.x / 2.0f, rect.position.y + rect.size.y / 2.0f});

	layerRender.draw(shape);
	layerRender.draw(text);
}

void visualLayer::setDots(const model::Neurons &newNeurons) {
	dots.net = newNeurons.net;
	dots.out = newNeurons.out;

	setUpdate();
}

void visualLayer::set_weights(const model::LayerParameters &Param) {
	parameters.set(Param);

	setUpdate();
}

void visualParamLayer::updateGrad(const model::LayerParameters &new_grad) {
	grad.set(new_grad);

	setUpdate();
}
} // namespace nn::visualizer
