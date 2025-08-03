#include "FnnVisualizer.hpp"
#include "../../visualizer/fonts.hpp"
#include "DenseLayer.hpp"
#include "network/IvisualNetwork.hpp"
#include "tensor.hpp"
#include <SFML/System/Vector2.hpp>
#include <cstddef>

namespace nn::visualizer::fnn {
FnnVisualier::FnnVisualier(
    const std::shared_ptr<StateManager> state_,
    const std::uint32_t width,
    const model::fnn::FNNConfig &_config)
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
    const global::Tensor &net,
    const global::Tensor &out,
    const model::fnn::LayerParams &parameters,
    const model::fnn::LayerParams &gradients) {
	float _width = visualWidth / Layers.size();
	float offset = _width * index;
	Layers[index] = std::make_unique<VisualDenseLayer>(
	    _width,
	    net,
	    out,
	    parameters,
	    gradients,
	    sf::Vector2f(offset, 0));
}

VisualDenseLayer::VisualDenseLayer(
    const std::uint32_t _width,
    const global::Tensor &net_,
    const global::Tensor &out_,
    const model::fnn::LayerParams &_parameters,
    const model::fnn::LayerParams &_gradients,
    const sf::Vector2f _pos)
    : net(net_),
      out(out_),
      parameters(_parameters),
      gradients(_gradients),
      pos(_pos),
      cacheNeurons(net.numElements()),
      cachePrevNeurons(_parameters.prevSize()),
      width(_width) {
	doCacheNeurons();
	doCacheWeights();
}

void VisualDenseLayer::doCacheWeights() {
	float scale = getScaleFactor(parameters.prevSize());
	float neuron_width_scaled = global::NEURON_WIDTH * scale;

	float gap = calculateGap(parameters.prevSize(), neuron_width_scaled);
	float x = pos.x;

	for (size_t neuron = 0; neuron < parameters.prevSize(); ++neuron) {
		float y = neuron * (gap + neuron_width_scaled);
		cachePrevNeurons[neuron] = sf::FloatRect(
		    sf::Vector2f(x, y),
		    sf::Vector2f(neuron_width_scaled, neuron_width_scaled));
	}
}

void VisualDenseLayer::doCacheNeurons() {
	float scale = getScaleFactor(net.numElements());
	float neuron_width_scaled = global::NEURON_WIDTH * scale;

	float gap = calculateGap(net.numElements(), neuron_width_scaled);
	float x = pos.x + width - neuron_width_scaled;

	for (size_t neuron = 0; neuron < net.numElements(); ++neuron) {
		float y = neuron * (gap + neuron_width_scaled);
		cacheNeurons[neuron] = sf::FloatRect(
		    sf::Vector2f(x, y),
		    sf::Vector2f(neuron_width_scaled, neuron_width_scaled));
	}
}

float VisualDenseLayer::getScaleFactor(std::size_t neuron_count) {
	float maxNeuronSpace = MODEL_HEIGHT - (neuron_count)*global::MIN_GAP;

	float neuronWidth = maxNeuronSpace / std::max<float>(neuron_count, 1);
	neuronWidth = std::clamp(neuronWidth, global::MIN_NEURON_WIDTH, global::MAX_NEURON_WIDTH);

	return neuronWidth / global::MAX_NEURON_WIDTH;
}

sf::Vector2f VisualDenseLayer::getCenter(const sf::FloatRect &rect) {
	return {rect.position.x, rect.position.y + rect.size.y / 2.f};
}

void VisualDenseLayer::drawWeights(const size_t neuron_i, sf::RenderTexture &target) {
	for (size_t neuronP = 0; neuronP < parameters.prevSize(); neuronP++) {
		sf::VertexArray line_(sf::PrimitiveType::LineStrip, 3);

		sf::Vector2f from = getCenter(cachePrevNeurons[neuronP]);
		sf::Vector2f to = getCenter(cacheNeurons[neuron_i]);
		sf::Vector2f mid = (from + to) / 2.f;

		line_[0].position = from;
		line_[1].position = mid;
		line_[2].position = to;

		line_[0].color = LINE_COLOR;
		line_[0].color.a = parameters.weights({neuron_i, neuronP}) * 50;
		line_[1].color = line_[0].color;
		line_[2].color = getColorFromTextT(getTextT(neuron_i, neuronP));
		target.draw(line_);
	}
}

int VisualDenseLayer::getParamCount() const {
	return parameters.size() * parameters.prevSize();
}

void VisualDenseLayer::drawGapWeight(sf::RenderTexture &target) {
	sf::VertexArray line_(sf::PrimitiveType::LineStrip, 2);

	line_[0].position = sf::Vector2f(0, MODEL_HEIGHT / 2.) + pos;
	line_[1].position = sf::Vector2f(width, MODEL_HEIGHT / 2.) + pos;

	line_[0].color = LINE_COLOR;
	line_[1].color = LINE_COLOR;
	target.draw(line_);
}

sf::Color VisualDenseLayer::getNeuronColor(const global::ValueType value) {
	sf::Color newColor = NEURON_BG_COLOR;
	newColor.b *= value;
	return newColor;
}

void VisualDenseLayer::drawNeuron(
    const sf::FloatRect &rect,
    const global::ValueType input,
    const global::ValueType output,
    sf::RenderTexture &target) {
	sf::RectangleShape shape(rect.size);
	shape.setFillColor(getNeuronColor(output));
	shape.setPosition(rect.position);

	target.draw(shape);

	if (10 * rect.size.y / global::NEURON_WIDTH > global::MIN_FONT_SIZE) {
		std::ostringstream ss;
		ss << std::fixed << std::setprecision(4) << input << "\n"
		   << output;

		sf::Text text(Fonts::getFont());
		text.setCharacterSize(10 * rect.size.y / global::NEURON_WIDTH);
		text.setString(ss.str());
		text.setFillColor(NEURON_TEXT_COLOR);

		sf::FloatRect textBounds = text.getLocalBounds();
		text.setOrigin({textBounds.position.x + textBounds.size.x / 2.0f,
		                textBounds.position.y + textBounds.size.y / 2.0f});

		text.setPosition({rect.position.x + rect.size.x / 2.0f, rect.position.y + rect.size.y / 2.0f});

		target.draw(text);
	}
}

void VisualDenseLayer::renderNeuron(const size_t index, sf::RenderTexture &target) {
	if (getParamCount() < MAX_WEIGHT_TO_RENDER) {
		drawWeights(index, target);
	}

	drawNeuron(cacheNeurons[index], net[index], out[index], target);
}

void VisualDenseLayer::drawNeurons(sf::RenderTexture &target) {
	if (getParamCount() >= MAX_WEIGHT_TO_RENDER) {
		drawGapWeight(target);
	}

	for (size_t neuron = 0; neuron < net.numElements(); ++neuron) {
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

textType VisualDenseLayer::getTextT(const size_t layer_i, const size_t layer_p) {
	if (gradients.weights({layer_i, layer_p}) < 0)
		return textType::DOWN;

	if (gradients.weights({layer_i, layer_p}) > 0)
		return textType::UP;

	return textType::NORMAL;
}
} // namespace nn::visualizer::fnn
