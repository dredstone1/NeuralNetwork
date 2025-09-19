#include "VisualizerRenderer.hpp"
#include "network/IvisualNetwork.hpp"
#include "tensor.hpp"
#include "visualModel.hpp"
#include <SFML/System/Vector2.hpp>
#include <memory>

namespace nn::visualizer {

constexpr std::uint32_t NN_WIDTH = 1055u;
VisualRender::VisualRender(std::shared_ptr<StateManager> vstate)
    : Vstate(vstate),
      winSize(getWinSize(Vstate->config.visualConfig.enableNetwrokVisual)),
      window(sf::VideoMode(winSize), WINDOW_TITLE.data()),
      interface(vstate),
      statusV(vstate),
      Vgraph(vstate) {
	if (Vstate->config.visualConfig.enableNetwrokVisual) {
		visualModel = std::make_unique<ModelPanel>(vstate);
	}
}

sf::Vector2u VisualRender::getWinSize(bool enableNetwork) {
	winSize = sf::Vector2u(WINDOW_WIDTH, WINDOW_HEIGHT);
	if (!enableNetwork) {
		winSize -= sf::Vector2u(MODEL_WIDTH + UI_GAP, 0);
	}
	return winSize;
}

void VisualRender::processEvents() {
	while (const std::optional event = window.pollEvent()) {
		if (event->is<sf::Event::Closed>()) {
			window.close();
		} else if (event->is<sf::Event::MouseButtonPressed>()) {
			interface.handleNoClick();
		} else if (event->is<sf::Event::MouseButtonReleased>()) {
			float x = UI_GAP;
			if (Vstate->config.visualConfig.enableNetwrokVisual) {
				x += NN_WIDTH + UI_GAP * 2;
			}

			interface.handleClick(sf::Mouse::getPosition(window), {x, UI_GAP});
		} else if (event->is<sf::Event::Resized>()) {
			need_resize = true;
		} else if (const auto *keyPressed = event->getIf<sf::Event::KeyPressed>()) {
			if (keyPressed->scancode == sf::Keyboard::Scancode::Space) {
				Vstate->toggle(SettingType::Pause);
				interface.setUpdate();
			}
		}
	}
}

void VisualRender::resetSize() {
	if (need_resize) {
		window.setSize(winSize);
	}

	need_resize = false;
}

void VisualRender::renderPanels() {
	float networkOffset = 0;
	if (visualModel) {
		visualModel->render();
		sf::Sprite visualNetworkSprite = visualModel->getSprite();
		visualNetworkSprite.setPosition({UI_GAP, UI_GAP});
		window.draw(visualNetworkSprite);
		networkOffset += visualNetworkSprite.getGlobalBounds().size.x + UI_GAP;
	}

	interface.render();
	sf::Sprite interfaceSprite = interface.getSprite();
	interfaceSprite.setPosition({networkOffset + UI_GAP, UI_GAP});
	window.draw(interfaceSprite);

	statusV.render();
	sf::Sprite statusSprite = statusV.getSprite();
	statusSprite.setPosition({networkOffset + UI_GAP, UI_GAP * 2 + VINTERFACE_HEIGHT});
	window.draw(statusSprite);

	Vgraph.render();
	sf::Sprite graphSprite = Vgraph.getSprite();
	graphSprite.setPosition({networkOffset + UI_GAP, UI_GAP * 3 + VINTERFACE_HEIGHT + VSTATUS_HEIGHT});
	window.draw(graphSprite);
}

void VisualRender::fullUpdate() {
	resetSize();

	statusV.setUpdate();
	interface.setUpdate();
	if (visualModel) {
		visualModel->setUpdate();
	}
	Vgraph.setUpdate();
}

void VisualRender::doFrame(int &frameCount, int &batchCount,
                           sf::Clock &fpsClock) {
	processEvents();

	if (fpsClock.getElapsedTime().asSeconds() >= 1.0f) {
		const float timeOffset = fpsClock.getElapsedTime().asSeconds();
		fps = frameCount / timeOffset;
		bps = (Vstate->currentBatch - batchCount) / timeOffset;

		fpsClock.restart();
		frameCount = 0;
		batchCount = Vstate->currentBatch;
		statusV.updateFps(fps);
		statusV.updateBps(bps);
		fullUpdate();
	}

	if (updateStatus()) {
		clear();
		frameCount++;

		renderPanels();
		window.display();
	}
}

void VisualRender::clear() {
	window.clear(BG_COLOR);
}

void VisualRender::renderLoop() {
	running.store(true);
	sf::Clock fpsClock;
	int frameCount = 0, batchCount = 0;

	window.setFramerateLimit(FPS_LIMIT);

	clear();
	while (window.isOpen() && running) {
		doFrame(frameCount, batchCount, fpsClock);
	}

	window.close();
}

void VisualRender::close() {
	running.store(false);
}

bool VisualRender::updateStatus() {
	return interface.updateStatus() ||
	       statusV.updateStatus() ||
	       (visualModel && visualModel->updateStatus());
}

void VisualRender::start() {
	running.store(true);
	renderLoop();
}

void VisualRender::updateLost(
    const global::ValueType newDataLost,
    int index) {
	Vgraph.addLostData(newDataLost, index);
}

void VisualRender::updateEvaluate(
    const global::ValueType newDataEvaluate,
    int index) {
	Vgraph.addEvaluateData(newDataEvaluate, index);
}

VisualRender::~VisualRender() {
	close();
}

void VisualRender::setNewPhaseMode(const NnMode nn_mode) {
	statusV.setUpdate();
	Vstate->nnMode.store(nn_mode);
}

void VisualRender::updatePrediction(const global::Prediction &pre) {
	if (visualModel) {
		visualModel->setPrediction(pre);
	}
}

void VisualRender::updateInput(const global::Tensor &input) {
	if (visualModel) {
		visualModel->setInput(input);
	}
}

void VisualRender::updateLearningRate(const global::ValueType lr) {
	statusV.updateLerningRate(lr);
}

void VisualRender::addVisualSubNetwork(
    const std::shared_ptr<IVisualNetwork> newVisual) {
	if (visualModel) {
		visualModel->addVisualSubNetwork(newVisual);
	}
}

void VisualRender::resetGraph() {
    Vgraph.reset();
}
} // namespace nn::visualizer
