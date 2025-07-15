#include "visualModel.hpp"
#include "../networks/fnn/FnnVisualizer.hpp"
#include "fonts.hpp"

namespace nn::visualizer {
DummyLayer::DummyLayer(const int size_, const sf::Vector2f pos_)
    : pos(pos_),
      layerRender({global::NEURON_WIDTH, MODEL_HEIGHT}),
      cacheNeurons(size_),
      values(size_, 0) {
	createVLayer();
}

void DummyLayer::createVLayer() {
	float scale = getScaleFactor(size());
	float neuron_width_scaled = global::NEURON_WIDTH * scale;

	float gap = calculateGap(size(), neuron_width_scaled);
	float x = pos.x + global::NEURON_WIDTH - neuron_width_scaled;

	for (int neuron = 0; neuron < size(); ++neuron) {
		float y = neuron * (gap + neuron_width_scaled);
		cacheNeurons[neuron] = sf::FloatRect(sf::Vector2f(x, y), {neuron_width_scaled, neuron_width_scaled});
	}

	pos.x = pos.x - x;
}

void DummyLayer::clear() {
	layerRender.clear(MODEL_BG);
}

void DummyLayer::setValues(const global::ParamMetrix &newValues) {
	values = newValues;
}

void DummyLayer::display() {
	layerRender.display();
}

void DummyLayer::draw(sf::RenderTexture &target) {
	for (int i = 0; i < size(); ++i) {
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

void DummyLayer::renderNeuron(sf::RenderTexture &target, const int index) {
	sf::RectangleShape shape(cacheNeurons[index].size);
	shape.setFillColor(getNeuronColor(values[index]));
	shape.setPosition(cacheNeurons[index].position + pos);

	target.draw(shape);

	if (10 * cacheNeurons[index].size.y / global::NEURON_WIDTH > global::MIN_FONT_SIZE) {
		std::ostringstream ss;
		ss << std::fixed << std::setprecision(4) << values[index];

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
      predictionLayer(state_->config.networkConfig.outputSize(), {MODEL_WIDTH - global::NEURON_WIDTH, 0}),
      inputLayer(state_->config.networkConfig.inputSize()),
      modelRender({MODEL_WIDTH, MODEL_HEIGHT}) {
}

sf::Sprite ModelPanel::getSprite() const {
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

	inputLayer.draw(modelRender);
	predictionLayer.draw(modelRender);

	renderSubNetworks();

	display();
}

float ModelPanel::getSubNetworkOffset(const int index) const {
	return SUB_NETWORKS_WIDTH * index / subNetworks.size();
}

void ModelPanel::renderSubNetworks() {
	for (size_t i = 0; i < subNetworks.size(); ++i) {
		renderSubNetwork(i);
	}
}

void ModelPanel::renderSubNetwork(const int index) {
	subNetworks[index]->render();

	sf::Sprite sub = subNetworks[index]->getSprite();
	sub.setPosition({global::NEURON_WIDTH + getSubNetworkOffset(index), 0});

	modelRender.draw(sub);
}

void ModelPanel::setPrediction(const global::Prediction &pre) {
	global::ParamMetrix output(predictionLayer.size(), 0);
	output[pre.index] = 1;
	predictionLayer.setValues(output);

	setUpdate();
}

void ModelPanel::setInput(const global::ParamMetrix &input) {
	inputLayer.setValues(input);

	setUpdate();
}

void ModelPanel::addVisualSubNetwork(const std::shared_ptr<IVisualNetwork> newVisual) {
	subNetworks.push_back(newVisual);
}
} // namespace nn::visualizer
