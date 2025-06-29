#include "visualModel.hpp"
#include "Globals.hpp"
#include "fonts.hpp"
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/System/Vector2.hpp>
#include <iostream>

namespace nn::visualizer {
DummyLayer::DummyLayer(const int size_, const std::shared_ptr<StateManager> state_)
    : Panel(state_),
      layerRender({NEURON_WIDTH, MODEL_HEIGHT}),
      cacheNeurons(size_),
      values(size_, 0) {
	createVLayer();
}

void DummyLayer::createVLayer() {
	float scale = getScaleFactor(size());
	float neuron_width_scaled = NEURON_WIDTH * scale;

	float gap = calculateGap(size(), neuron_width_scaled);

	for (int neuron = 0; neuron < size(); neuron++) {
		float y = neuron * (gap + neuron_width_scaled);
		cacheNeurons[neuron] = sf::FloatRect({0, y}, {neuron_width_scaled, neuron_width_scaled});
	}
}

sf::Sprite DummyLayer::getSprite() {
	return sf::Sprite(layerRender.getTexture());
}

void DummyLayer::clear() {
	layerRender.clear(MODEL_BG);
}

void DummyLayer::setValues(const global::ParamMetrix &newValues) {
	values = newValues;

	setUpdate();
}

void DummyLayer::display() {
	layerRender.display();
}

void DummyLayer::renderLayer() {
	for (int i = 0; i < size(); i++) {
		renderNeuron(i);
	}
}

void DummyLayer::doRender() {
	clear();

	renderLayer();

	display();
}

float DummyLayer::getScaleFactor(std::size_t neuron_count) {
	float maxNeuronSpace = MODEL_HEIGHT - (neuron_count)*MIN_GAP;

	float neuronWidth = maxNeuronSpace / std::max<float>(neuron_count, 1);
	neuronWidth = std::clamp(neuronWidth, MIN_NEURON_WIDTH, MAX_NEURON_WIDTH);

	return neuronWidth / MAX_NEURON_WIDTH;
}

float DummyLayer::calculateGap(const int size, const float scale) {
	return (MODEL_HEIGHT - (size * scale)) / (size - 1);
}

sf::Color DummyLayer::getNeuronColor(const global::ValueType value) {
	sf::Color newColor = NEURON_BG_COLOR;
	newColor.b *= value;
	return newColor;
}

void DummyLayer::renderNeuron(const int index) {
	sf::RectangleShape shape({NEURON_WIDTH, NEURON_WIDTH});
	shape.setFillColor(getNeuronColor(values[index]));
	shape.setPosition(cacheNeurons[index].position);

	std::ostringstream ss;
	ss << std::fixed << std::setprecision(4) << values[index];

	sf::Text text(Fonts::getFont());
	text.setCharacterSize(10 * cacheNeurons[index].size.y / NEURON_WIDTH);
	text.setString(ss.str());
	text.setFillColor(NEURON_TEXT_COLOR);

	sf::FloatRect textBounds = text.getLocalBounds();
	text.setOrigin({textBounds.position.x + textBounds.size.x / 2.0f,
	                textBounds.position.y + textBounds.size.y / 2.0f});
	text.setPosition({cacheNeurons[index].position.x + cacheNeurons[index].size.x / 2.0f, cacheNeurons[index].position.y + cacheNeurons[index].size.y / 2.0f});

	layerRender.draw(shape);
	layerRender.draw(text);
}

ModelPanel::ModelPanel(const std::shared_ptr<StateManager> state_)
    : Panel(state_),
      predictionLayer(state_->config.networkConfig.outputSize(), state_),
      inputLayer(state_->config.networkConfig.inputSize(), state_),
      modelRender({MODEL_WIDTH, MODEL_HEIGHT}) {
	createVModel();
}

void ModelPanel::createVModel() {
}

sf::Sprite ModelPanel::getSprite() {
	return sf::Sprite(modelRender.getTexture());
}

void ModelPanel::clear() {
	modelRender.clear(MODEL_BG);
}

void ModelPanel::display() {
	modelRender.display();
}

void ModelPanel::doRender() {
	clear();

	inputLayer.render();
	sf::Sprite input = inputLayer.getSprite();
	input.setPosition({0, 0});
	modelRender.draw(input);

	predictionLayer.render();
	sf::Sprite prediction = predictionLayer.getSprite();
	prediction.setPosition({MODEL_WIDTH - NEURON_WIDTH, 0});
	modelRender.draw(prediction);

	display();
}

void ModelPanel::setPrediction(const int index) {
	global::ParamMetrix prediction(predictionLayer.size(), 0);
	prediction[index] = 1;
	predictionLayer.setValues(prediction);

	setUpdate();
}

void ModelPanel::setInput(const global::ParamMetrix &input) {
	inputLayer.setValues(input);

	setUpdate();
}
} // namespace nn::visualizer
