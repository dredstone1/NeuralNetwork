#include "visualModel.hpp"
#include "../networks/fnn/FnnVisualizer.hpp"
#include "Globals.hpp"
#include "fonts.hpp"
#include "network/IvisualNetwork.hpp"
#include <SFML/Graphics/Color.hpp>
#include <SFML/System/Vector2.hpp>
#include <cstdint>
#include <memory>

namespace nn::visualizer {
DummyLayer::DummyLayer(const size_t size_, const sf::Vector2f pos_)
    : pos(pos_),
      layerRender({global::NEURON_WIDTH, MODEL_HEIGHT}),
      cacheNeurons(size_),
      values({size_}) {
	createVLayer();
}

void DummyLayer::createVLayer() {
	float scale = getScaleFactor(size());
	float neuron_width_scaled = global::NEURON_WIDTH * scale;

	float gap = calculateGap(size(), neuron_width_scaled);
	float x = pos.x + global::NEURON_WIDTH - neuron_width_scaled;

	for (size_t neuron = 0; neuron < size(); ++neuron) {
		float y = neuron * (gap + neuron_width_scaled);
		cacheNeurons[neuron] = sf::FloatRect(sf::Vector2f(x, y), {neuron_width_scaled, neuron_width_scaled});
	}

	pos.x = pos.x - x;
}

void DummyLayer::clear() {
	layerRender.clear(MODEL_BG);
}

void DummyLayer::setPos(const sf::Vector2f &newPos) {
	pos = newPos;
	createVLayer();
}

void DummyLayer::setValues(const global::Tensor &newValues) {
	values = newValues;
}

void DummyLayer::display() {
	layerRender.display();
}

void DummyLayer::draw(sf::RenderTexture &target) {
	for (size_t i = 0; i < size(); ++i) {
		renderNeuron(target, i);
	}
}

float DummyLayer::getScaleFactor(std::size_t neuron_count) {
	float maxNeuronSpace = MODEL_HEIGHT - (neuron_count)*global::MIN_GAP;

	float neuronWidth = maxNeuronSpace / std::max<float>(neuron_count, 1);
	neuronWidth = std::clamp(neuronWidth, global::MIN_NEURON_WIDTH, global::MAX_NEURON_WIDTH);

	return neuronWidth / global::MAX_NEURON_WIDTH;
}

float DummyLayer::calculateGap(const int size, const float scale) {
	return (MODEL_HEIGHT - (size * scale)) / (size - 1);
}

sf::Color DummyLayer::getNeuronColor(const global::ValueType value) {
	sf::Color newColor = fnn::NEURON_BG_COLOR;
	newColor.b *= value;
	return newColor;
}

void DummyLayer::renderNeuron(sf::RenderTexture &target, const size_t index) {
	sf::RectangleShape shape(cacheNeurons[index].size);
	shape.setFillColor(getNeuronColor(values({index})));
	shape.setPosition(cacheNeurons[index].position + pos);

	target.draw(shape);

	if (10 * cacheNeurons[index].size.y / global::NEURON_WIDTH > global::MIN_FONT_SIZE) {
		std::ostringstream ss;
		ss << std::fixed << std::setprecision(4) << values({index});

		sf::Text text(Fonts::getFont());
		text.setCharacterSize(10 * cacheNeurons[index].size.y / global::NEURON_WIDTH);
		text.setString(ss.str());
		text.setFillColor(fnn::NEURON_TEXT_COLOR);

		sf::FloatRect textBounds = text.getLocalBounds();
		text.setOrigin({textBounds.position.x + textBounds.size.x / 2.0f,
		                textBounds.position.y + textBounds.size.y / 2.0f});
		text.setPosition(sf::Vector2f(cacheNeurons[index].position.x + cacheNeurons[index].size.x / 2.0f, cacheNeurons[index].position.y + cacheNeurons[index].size.y / 2.0f) + pos);
		target.draw(text);
	}
}

ModelPanel::ModelPanel(const std::shared_ptr<StateManager> state_)
    : Panel(state_),
      predictionLayer(state_->config.networkConfig.outputSize()),
      inputLayer(state_->config.networkConfig.inputSize()),
      modelRender({MODEL_WIDTH, MODEL_HEIGHT}) {
	predictionLayer.setPos({MODEL_WIDTH - (float)predictionLayer.getWidth(), 0});

	std::uint32_t offset = MODEL_WIDTH;

	offset -= predictionLayer.getWidth();
	offset -= inputLayer.getWidth();

	subNetworksRect.size.x = offset;
	subNetworksRect.size.y = MODEL_HEIGHT;
	subNetworksRect.position.x = inputLayer.getWidth();
}

sf::Sprite ModelPanel::getSprite() const {
	return sf::Sprite(modelRender.getTexture());
}

void ModelPanel::clear() {
	modelRender.clear(sf::Color::Blue);
}

void ModelPanel::display() {
	modelRender.display();
}

void ModelPanel::doRender() {
	clear();

	inputLayer.draw(modelRender);
	predictionLayer.draw(modelRender);

	renderSubNetworks();

	display();
}

float ModelPanel::getSubNetworkOffset(const int index) const {
	return subNetworksRect.size.x * index / subNetworks.size();
}

void ModelPanel::renderSubNetworks() {
	for (size_t i = 0; i < subNetworks.size(); ++i) {
		renderSubNetwork(i);
	}
}

void ModelPanel::renderSubNetwork(const size_t index) {
	subNetworks[index]->render();

	sf::Sprite sub = subNetworks[index]->getSprite();
	sub.setPosition({subNetworksRect.position.x + getSubNetworkOffset(index), 0});

	modelRender.draw(sub);
}

void ModelPanel::setPrediction(const global::Prediction &pre) {
	global::Tensor output({predictionLayer.size()});
	output({pre.index}) = 1;
	predictionLayer.setValues(output);

	setUpdate();
}

void ModelPanel::setInput(const global::Tensor &input) {
	inputLayer.setValues(input);

	setUpdate();
}

void ModelPanel::addVisualSubNetwork(const std::shared_ptr<IVisualNetwork> newVisual) {
	subNetworks.push_back(newVisual);

	for (auto &sub : subNetworks) {
		sub->setWidth(subNetworksRect.size.x / subNetworks.size());
	}
}

bool ModelPanel::updateStatus() const {
	for (size_t i = 0; i < subNetworks.size(); ++i) {
		if (subNetworks[i]->updateStatus()) {
			return true;
		}
	}

	return false;
}
} // namespace nn::visualizer
