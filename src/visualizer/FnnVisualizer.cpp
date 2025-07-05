#include "FnnVisualizer.hpp"
#include "IvisualNetwork.hpp"
#include "fonts.hpp"
#include <SFML/System/Vector2.hpp>

namespace nn::visualizer {
FnnVisualier::FnnVisualier(
    const std::shared_ptr<StateManager> state_,
    const std::uint32_t width,
    const model::FNNConfig &_config)
    : IVisualNetwork(state_, width),
      config(_config),
      Layers(config.layersConfig.size() + 1) {
}

void FnnVisualier::renderNetwork() {
	renderLayers();
}

void FnnVisualier::renderLayers() {
	for (size_t i = 0; i < Layers.size(); ++i) {
		renderLayer(i);
	}
}

void FnnVisualier::renderLayer(const int index) {
	if (!Layers[index])
		return;

	Layers[index]->draw(networkRender);
}

void FnnVisualier::initLayer(
    const int index,
    const model::Neurons &dots,
    const model::LayerParameters &parameters,
    const model::LayerParameters &gradients) {
	float _width = visualWidth / Layers.size();
	float offset = _width * index;
	Layers[index] = std::make_unique<VisualDenseLayer>(_width, dots, parameters, gradients, sf::Vector2f(offset, 0));
}

VisualDenseLayer::VisualDenseLayer(
    const std::uint32_t _width,
    const model::Neurons &_dots,
    const model::LayerParameters &_parameters,
    const model::LayerParameters &_gradients,
    const sf::Vector2f _pos)
    : dots(_dots),
      parameters(_parameters),
      gradients(_gradients),
      pos(_pos),
      cacheNeurons(_dots.size()),
      cachePrevNeurons(_parameters.getPrevSize()),
      width(_width) {
	doCacheNeurons();
	doCacheWeights();
}

void VisualDenseLayer::doCacheWeights() {
	float scale = getScaleFactor(parameters.getPrevSize());
	float neuron_width_scaled = NEURON_WIDTH * scale;

	float gap = calculateGap(parameters.getPrevSize(), neuron_width_scaled);
	float x = pos.x;

	for (size_t neuron = 0; neuron < parameters.getPrevSize(); ++neuron) {
		float y = neuron * (gap + neuron_width_scaled);
		cachePrevNeurons[neuron] = sf::FloatRect(sf::Vector2f(x, y), {neuron_width_scaled, neuron_width_scaled});
	}
}

void VisualDenseLayer::doCacheNeurons() {
	float scale = getScaleFactor(dots.size());
	float neuron_width_scaled = NEURON_WIDTH * scale;

	float gap = calculateGap(dots.size(), neuron_width_scaled);
	float x = pos.x + width - neuron_width_scaled;

	for (size_t neuron = 0; neuron < dots.size(); ++neuron) {
		float y = neuron * (gap + neuron_width_scaled);
		cacheNeurons[neuron] = sf::FloatRect(sf::Vector2f(x, y), {neuron_width_scaled, neuron_width_scaled});
	}
}

float VisualDenseLayer::getScaleFactor(std::size_t neuron_count) {
	float maxNeuronSpace = MODEL_HEIGHT - (neuron_count)*MIN_GAP;

	float neuronWidth = maxNeuronSpace / std::max<float>(neuron_count, 1);
	neuronWidth = std::clamp(neuronWidth, MIN_NEURON_WIDTH, MAX_NEURON_WIDTH);

	return neuronWidth / MAX_NEURON_WIDTH;
}

sf::Vector2f VisualDenseLayer::getCenter(const sf::FloatRect &rect) {
	return {rect.position.x, rect.position.y + rect.size.y / 2.f};
}

void VisualDenseLayer::drawWeights(const int neuron_i, sf::RenderTexture &target) {
	for (size_t neuronP = 0; neuronP < parameters.getPrevSize(); neuronP++) {
		sf::VertexArray line_(sf::PrimitiveType::LineStrip, 3);

		sf::Vector2f from = getCenter(cachePrevNeurons[neuronP]);
		sf::Vector2f to = getCenter(cacheNeurons[neuron_i]);
		sf::Vector2f mid = (from + to) / 2.f;

		line_[0].position = from;
		line_[1].position = mid;
		line_[2].position = to;

		line_[0].color = LINE_COLOR;
		line_[0].color.a = parameters.weights[neuron_i][neuronP] * 50;
		line_[1].color = line_[0].color;
		line_[2].color = getColorFromTextT(getTextT(neuron_i, neuronP));
		target.draw(line_);
	}
}

sf::Color VisualDenseLayer::getNeuronColor(const global::ValueType value) {
	sf::Color newColor = NEURON_BG_COLOR;
	newColor.b *= value;
	return newColor;
}

void VisualDenseLayer::drawNeuron(const sf::FloatRect &rect, const double input, const double output, sf::RenderTexture &target) {
	sf::RectangleShape shape(rect.size);
	shape.setFillColor(getNeuronColor(output));
	shape.setPosition(rect.position);

	std::ostringstream ss;
	ss << std::fixed << std::setprecision(4) << input << "\n"
	   << output;

	sf::Text text(Fonts::getFont());
	text.setCharacterSize(10 * rect.size.y / NEURON_WIDTH);
	text.setString(ss.str());
	text.setFillColor(NEURON_TEXT_COLOR);

	sf::FloatRect textBounds = text.getLocalBounds();
	text.setOrigin({textBounds.position.x + textBounds.size.x / 2.0f,
	                textBounds.position.y + textBounds.size.y / 2.0f});

	text.setPosition({rect.position.x + rect.size.x / 2.0f, rect.position.y + rect.size.y / 2.0f});

	target.draw(shape);
	target.draw(text);
}

void VisualDenseLayer::renderNeuron(const int index, sf::RenderTexture &target) {
	drawWeights(index, target);
	drawNeuron(cacheNeurons[index], dots.net[index], dots.out[index], target);
}

void VisualDenseLayer::drawNeurons(sf::RenderTexture &target) {
	for (size_t neuron = 0; neuron < dots.size(); ++neuron) {
		renderNeuron(neuron, target);
	}
}

void VisualDenseLayer::draw(sf::RenderTexture &target) {
	drawNeurons(target);
}

float VisualDenseLayer::calculateDistance(const sf::Vector2f pos1, const sf::Vector2f pos2) {
	return sqrt(pow(pos1.x - pos2.x, 2) + pow(pos1.y - pos2.y, 2));
}
sf::Angle VisualDenseLayer::calculateAngle(const sf::Vector2f pos1, const sf::Vector2f pos2) {
	return sf::radians(atan2(pos2.y - pos1.y, pos2.x - pos1.x));
}

sf::Color VisualDenseLayer::getColorFromTextT(const textType text_type) {
	return color_lookup[static_cast<size_t>(text_type)];
}

float VisualDenseLayer::calculateGap(const int size, const float scale) {
	if (size <= 1)
		return 0.f;

	return (MODEL_HEIGHT - (size * scale)) / (size - 1);
}

textType VisualDenseLayer::getTextT(const int layer_i, const int layer_p) {
	if (gradients.weights[layer_i][layer_p] < 0)
		return textType::DOWN;

	if (gradients.weights[layer_i][layer_p] > 0)
		return textType::UP;

	return textType::NORMAL;
}
} // namespace nn::visualizer
