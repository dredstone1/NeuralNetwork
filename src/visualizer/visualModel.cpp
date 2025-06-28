#include "visualModel.hpp"

namespace nn::visualizer {
ModelPanel::ModelPanel(const std::shared_ptr<StateManager> state_)
    : Panel(state_),
      modelRender({MODEL_WIDTH, MODEL_HEIGHT}) {
	createVInterface();
}

void ModelPanel::createVInterface() {
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
	display();
}
} // namespace nn::visualizer
