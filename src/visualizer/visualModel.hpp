#ifndef VISUALMODEL
#define VISUALMODEL

#include "IvisualNetwork.hpp"
#include <SFML/Graphics.hpp>

namespace nn::visualizer {

struct neuronValues {
	global::ValueType net;
	global::ValueType out;
};

class DummyLayer {
  private:
	sf::Vector2f pos;

	void clear();
	void display();
	void createVLayer();

	static float getScaleFactor(const std::size_t neuron_count);
	static float calculateGap(const int size, const float scale);
	static sf::Color getNeuronColor(const global::ValueType value);

	void renderNeuron(sf::RenderTexture &target, const int index);

	void doCacheNeurons();

	sf::RenderTexture layerRender;
	std::vector<sf::FloatRect> cacheNeurons;
	global::ParamMetrix values;

  public:
	DummyLayer(const int size, const sf::Vector2f pos = {0, 0});
	~DummyLayer() = default;

	int size() const { return values.size(); }
	void setValues(const global::ParamMetrix &newValues);

	void draw(sf::RenderTexture &target);
};

class ModelPanel : public Panel {
  private:
	void clear();
	void display();

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

	void setPrediction(const global::Prediction &index);
	void setInput(const global::ParamMetrix &input);
	sf::Sprite getSprite() const;

	void addVisualSubNetwork(const std::shared_ptr<IVisualNetwork> newVisual);
};
} // namespace nn::visualizer

#endif // VISUALMODEL
