#ifndef VISUALMODEL
#define VISUALMODEL

#include "tensor.hpp"
#include <SFML/System/Vector2.hpp>
#include <cstdint>
#include <network/IvisualNetwork.hpp>

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

	void renderNeuron(sf::RenderTexture &target, const size_t index);

	void doCacheNeurons();

	sf::RenderTexture layerRender;
	std::vector<sf::FloatRect> cacheNeurons;
	global::Tensor values;

  public:
	DummyLayer(const size_t size, const sf::Vector2f pos = {0, 0});
	~DummyLayer() = default;

	size_t size() const { return values.numElements(); }
	void setValues(const global::Tensor &newValues);

	std::uint32_t getWidth() { return cacheNeurons[0].size.x; }
	void setPos(const sf::Vector2f &newPos);
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
	void renderSubNetwork(const size_t index);

	sf::FloatRect subNetworksRect;

  public:
	ModelPanel(const std::shared_ptr<StateManager> state_);
	~ModelPanel() = default;

	void setPrediction(const global::Prediction &index);
	void setInput(const global::Tensor &input);
	sf::Sprite getSprite() const;

	void addVisualSubNetwork(const std::shared_ptr<IVisualNetwork> newVisual);
	bool updateStatus() const override;

	std::uint32_t getsubNetworkWidth() const;
};
} // namespace nn::visualizer

#endif // VISUALMODEL
