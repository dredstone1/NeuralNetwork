#ifndef FNNVISUALNETWORK
#define FNNVISUALNETWORK

#include "DenseLayer.hpp"
#include "tensor.hpp"
#include <cstddef>
#include <network/IvisualNetwork.hpp>

namespace nn::visualizer::fnn {
constexpr sf::Color NEURON_TEXT_COLOR(255, 255, 255);
constexpr sf::Color NEURON_BG_COLOR(0, 0, 100);

enum class textType {
	UP,
	DOWN,
	NORMAL,
};

constexpr sf::Color FONT_COLOR_DOWN(255, 0, 0);
constexpr sf::Color FONT_COLOR_UP(0, 0, 255);
constexpr sf::Color FONT_COLOR_NORMAL(50, 50, 50);

constexpr sf::Color LINE_COLOR(0, 0, 0);

constexpr int MAX_WEIGHT_TO_RENDER = 1000;

static const std::array<sf::Color, 3> color_lookup = {
    FONT_COLOR_UP,
    FONT_COLOR_DOWN,
    FONT_COLOR_NORMAL,
};

class VisualDenseLayer {
  private:
	global::Tensor net{{0}};
	global::Tensor out{{0}};

	model::fnn::LayerParams parameters{0, 0};
	model::fnn::LayerParams gradients{0, 0};

	sf::Vector2f pos;

	std::vector<sf::FloatRect> cacheNeurons;
	std::vector<sf::FloatRect> cachePrevNeurons;

	void drawWeights(const size_t neuron_i, sf::RenderTexture &target);
	void drawGapWeight(sf::RenderTexture &target);
	void drawNeurons(sf::RenderTexture &target);
	void renderNeuron(const size_t index, sf::RenderTexture &target);
	void drawNeuron(
	    const sf::FloatRect &rect,
	    const global::ValueType input,
	    const global::ValueType output,
	    sf::RenderTexture &target);

	textType getTextT(const size_t layer_i, const size_t layer_p);
	sf::Color getColorFromTextT(const textType text_type);
	sf::Color getNeuronColor(const global::ValueType value);
	float getScaleFactor(const std::size_t neuron_count);
	float calculateDistance(const sf::Vector2f pos1, const sf::Vector2f pos2);
	sf::Angle calculateAngle(const sf::Vector2f pos1, const sf::Vector2f pos2);
	float calculateGap(const int size, const float scale);
	sf::Vector2f getCenter(const sf::FloatRect &rect);

	int getParamCount() const;

	void doCacheWeights();
	void doCacheNeurons();
	std::uint32_t width;

  public:
	VisualDenseLayer(
	    const std::uint32_t width,
	    const sf::Vector2f pos);	
    VisualDenseLayer() {}
	~VisualDenseLayer() = default;

	void draw(sf::RenderTexture &target);

	void setNet(const global::Tensor &newNet);
	void setOut(const global::Tensor &newOut);
	void setGrad(const model::fnn::LayerParams &newGrad);
	void setParam(const model::fnn::LayerParams &newParam);

	void setWidth(const std::uint32_t newWidth) { width = newWidth; }
	void setPos(const sf::Vector2f &newPos) { pos = newPos; }
};

class FnnVisualier : public IVisualNetwork {
  private:
	const model::fnn::FNNConfig &config;

	std::vector<VisualDenseLayer> Layers;

	void renderNetwork() override;
	void renderLayers();
	void renderLayer(const int index);

  public:
	FnnVisualier(
	    const std::shared_ptr<StateManager> state_,
	    const std::uint32_t width,
	    const model::fnn::FNNConfig &_config);
	~FnnVisualier() = default;

	void setNet(const size_t i, const global::Tensor &newNet);
	void setOut(const size_t i, const global::Tensor &newOut);
	void setGrad(const size_t i, const model::fnn::LayerParams &newGrad);
	void setParam(const size_t i, const model::fnn::LayerParams &newParam);
};
} // namespace nn::visualizer::fnn

#endif // FNNVISUALNETWORK
