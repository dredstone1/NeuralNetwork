#include "visualL.hpp"
#include "fonts.hpp"
#include <SFML/System/Angle.hpp>

namespace nn::visualizer {
visualL::visualL(const int _size, const int _prev_size, const std::shared_ptr<StateManager> state_, const std::uint32_t width)
    : Layer(_size, _prev_size, 0),
      Panel(state_),
      WIDTH(width),
      layerRender({width, NN_WIDTH}) {}

void visualL::display() {
	layerRender.display();
}

void visualL::clear() {
	layerRender.clear(sf::Color::Transparent);
}

void visualL::doRender() {
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

sf::Angle visualL::calculateAngle(const sf::Vector2f pos1, const sf::Vector2f pos2) {
	return sf::radians(atan2(pos2.y - pos1.y, pos2.x - pos1.x));
}

float visualL::getScaleFactor(std::size_t neuron_count) const {
	float totalHeight = static_cast<float>(NN_HEIGHT);
	float maxNeuronSpace = totalHeight - (neuron_count + 1) * MIN_GAP;

	float neuronWidth = maxNeuronSpace / std::max<float>(neuron_count, 1);
	neuronWidth = std::clamp(neuronWidth, MIN_NEURON_WIDTH, MAX_NEURON_WIDTH);

	return neuronWidth / MAX_NEURON_WIDTH;
}

void VParamLayer::drawWeights(const int neuron_i, const sf::Vector2f pos, const float prevGap, float scale) {
	const float HORIZONTAL_SHIFT_PER_WEIGHT_TEXT = 4.0f * scale;

	for (size_t neuronP = 0; neuronP < getPrevSize(); neuronP++) {
		float weightValue = parameters.weights[neuron_i][neuronP];

		float xP = 0.f;
		float neuron_width_scaled = NEURON_WIDTH * scale;
		float yP = prevGap + neuronP * (prevGap + neuron_width_scaled);

		sf::Vector2f prevNeuronTopLeft(xP, yP);

		float lineLength = calculateDistance(prevNeuronTopLeft, pos);
		sf::Angle angleDeg = calculateAngle(prevNeuronTopLeft, pos);

		float line_thickness_arg = std::max(std::min(weightValue * scale, 4.f * scale), 0.2f * scale);

		sf::RectangleShape line({lineLength, line_thickness_arg});
		line.setFillColor(sf::Color::Black);

		sf::Vector2f lineGraphicalOrigin(xP, yP + (neuron_width_scaled / 2.0f));
		line.setPosition(lineGraphicalOrigin);
		line.setRotation(angleDeg);

		std::ostringstream ss;
		ss << std::fixed << std::setprecision(4) << weightValue;

		sf::Text text(Fonts::getFont());
		text.setCharacterSize(static_cast<unsigned int>(10 * scale));
		text.setString(ss.str());
		text.setFillColor(getColorFromTextT(getTextT(neuron_i, neuronP)));

		sf::FloatRect textBounds = text.getLocalBounds();
		text.setOrigin({textBounds.position.x + textBounds.size.x / 2.0f,
		                textBounds.position.y + textBounds.size.y / 2.0f});

		float text_anchor_local_x = FRACTION_ALONG_LINE * lineLength;
		float text_anchor_local_y = line_thickness_arg / 2.0f;

		float cosA = cosf(angleDeg.asRadians());
		float sinA = sinf(angleDeg.asRadians());

		float text_pos_x_transformed = lineGraphicalOrigin.x + text_anchor_local_x * cosA - text_anchor_local_y * sinA;
		float text_pos_y_transformed = lineGraphicalOrigin.y + text_anchor_local_x * sinA + text_anchor_local_y * cosA;

		float final_text_pos_x = text_pos_x_transformed - (neuronP * HORIZONTAL_SHIFT_PER_WEIGHT_TEXT);
		float final_text_pos_y = text_pos_y_transformed;

		text.setPosition({final_text_pos_x, final_text_pos_y});
		text.setRotation(angleDeg);

		layerRender.draw(line);
		layerRender.draw(text);
	}
}

sf::Color VParamLayer::getColorFromTextT(const textType text_type) {
	return color_lookup[static_cast<size_t>(text_type)];
}

void visualL::drawNeurons() {
	float scale = getScaleFactor(getSize());
	float neuron_width_scaled = NEURON_WIDTH * scale;

	float gap = (NN_HEIGHT - (getSize() * neuron_width_scaled)) / (getSize() + 1);
	float prevGap = (NN_HEIGHT - (getPrevSize() * neuron_width_scaled)) / (getPrevSize() + 1);

	for (size_t neuron = 0; neuron < getSize(); neuron++) {
		renderNeuron(neuron, gap, prevGap, scale);
	}
}

void VParamLayer::renderNeuron(const int index, const float gap, const float prevGap, const float scale) {
	float neuron_width_scaled = NEURON_WIDTH * scale;
	float x = WIDTH - neuron_width_scaled;
	float y = gap + index * (gap + neuron_width_scaled);

	drawWeights(index, {x, y}, prevGap, scale);
	drawNeuron(dots.net[index], dots.out[index], {x, y}, scale);
}

void VEmptyLayer::renderNeuron(const int index, const float gap, const float, const float scale) {
	float neuron_width_scaled = NEURON_WIDTH * scale;
	float x = WIDTH - neuron_width_scaled;
	float y = gap + index * (gap + neuron_width_scaled);

	drawNeuron(dots.net[index], dots.out[index], {x, y}, scale);
}

textType VParamLayer::getTextT(const int layer_i, const int layer_p) {
	if (grad.weights[layer_i][layer_p] < 0)
		return textType::DOWN;
	if (grad.weights[layer_i][layer_p] > 0)
		return textType::UP;
	return textType::NORMAL;
}

textType VEmptyLayer::getTextT(const int, const int) {
	return textType::NORMAL;
}

textType visualL::getTextT(const int, const int) {
	return textType::NORMAL;
}

void visualL::drawNeuron(const double input, const double output, const sf::Vector2f pos, float scale) {
	float neuron_width_scaled = NEURON_WIDTH * scale;
	sf::RectangleShape shape({neuron_width_scaled, neuron_width_scaled});
	shape.setFillColor(sf::Color(0, 0, 100 * output));
	shape.setPosition(pos);

	std::ostringstream ss;
	ss << std::fixed << std::setprecision(4) << input << "\n"
	   << output;

	sf::Text text(Fonts::getFont());
	text.setCharacterSize(static_cast<unsigned int>(10 * scale));
	text.setString(ss.str());
	text.setFillColor(sf::Color::White);

	sf::FloatRect textBounds = text.getLocalBounds();
	text.setOrigin({textBounds.position.x + textBounds.size.x / 2.0f,
	                textBounds.position.y + textBounds.size.y / 2.0f});

	text.setPosition({pos.x + neuron_width_scaled / 2.0f, pos.y + neuron_width_scaled / 2.0f});

	layerRender.draw(shape);
	layerRender.draw(text);
}

void visualL::setDots(const model::Neurons &newNeurons) {
	dots.net = newNeurons.net;
	dots.out = newNeurons.out;

	setUpdate();
}

void visualL::set_weights(const model::LayerParameters &Param) {
	parameters.set(Param);

	setUpdate();
}

void VParamLayer::updateGrad(const model::LayerParameters &new_grad) {
	grad.set(new_grad);

	setUpdate();
}
} // namespace nn::visualizer
