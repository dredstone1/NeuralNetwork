#ifndef VISUALMODEL
#define VISUALMODEL

#include "IvisualNetwork.hpp"
#include "panel.hpp"
#include <Globals.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/System/Vector2.hpp>
#include <memory>
#include <vector>

namespace nn::visualizer {


constexpr float MIN_NEURON_WIDTH = 6.0f;
constexpr float MAX_NEURON_WIDTH = NEURON_WIDTH;
constexpr float MIN_GAP = 2.0f;

constexpr sf::Color NEURON_TEXT_COLOR(255, 255, 255);
constexpr sf::Color NEURON_BG_COLOR(0, 0, 100);


struct neuronValues {
	global::ValueType net;
	global::ValueType out;
};

class DummyLayer : public Panel {
  private:
	void clear();
	void display();
	void createVLayer();

	static float getScaleFactor(const std::size_t neuron_count);
	static float calculateGap(const int size, const float scale);
	static sf::Color getNeuronColor(const global::ValueType value);

	void doRender() override;

	void renderNeuron(const int index);
	void renderLayer();

	sf::RenderTexture layerRender;
	std::vector<sf::FloatRect> cacheNeurons;
	global::ParamMetrix values;

  public:
	DummyLayer(const int size, std::shared_ptr<StateManager> state_);
	~DummyLayer() = default;

	int size() const { return values.size(); }
	void setValues(const global::ParamMetrix &newValues);

	sf::Sprite getSprite();
};

class ModelPanel : public Panel {
  private:
	void clear();
	void display();
	void createVModel();

	void doRender() override;

	DummyLayer predictionLayer;
	DummyLayer inputLayer;
	sf::RenderTexture modelRender;

	float getSubNetworkOffset(const int index) const;

	std::vector<std::shared_ptr<IVisualNetwork>> subNetworks;

	void renderSubNetworks();
	void renderSubNetwork(const int index);

  public:
	ModelPanel(const std::shared_ptr<StateManager> state_);
	~ModelPanel() = default;

	void setPrediction(const int index);
	void setInput(const global::ParamMetrix &input);
	sf::Sprite getSprite();

	void addVisualSubNetwork(const std::shared_ptr<IVisualNetwork> newVisual);
};
} // namespace nn::visualizer

#endif // VISUALMODEL
