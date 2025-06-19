#ifndef VISUALL
#define VISUALL

#include "../model/layer.hpp"
#include "Globals.hpp"
#include "panel.hpp"
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/Rect.hpp>
#include <SFML/System/Angle.hpp>
#include <SFML/System/Vector2.hpp>
#include <vector>

namespace nn::visualizer {
constexpr std::uint32_t NN_HEIGHT = 770u;
constexpr std::uint32_t NN_WIDTH = 1055u;

constexpr std::uint32_t NEURON_RADIUS = 20;
constexpr std::uint32_t NEURON_WIDTH = NEURON_RADIUS * 2;

constexpr std::uint32_t calculate_width(const int layer_amount) {
	return (NN_WIDTH - NEURON_WIDTH * 2) / layer_amount;
}

constexpr sf::Color NORMAL_BG_LAYER(255, 255, 255);
constexpr sf::Color ACTIVE_BG_LAYER(187, 187, 187);

constexpr sf::Color FONT_COLOR_DOWN(255, 0, 0);
constexpr sf::Color FONT_COLOR_UP(0, 0, 255);
constexpr sf::Color FONT_COLOR_NORMAL(50, 50, 50);

constexpr sf::Color LINE_COLOR(0, 0, 0);
constexpr sf::Color NEURON_TEXT_COLOR(255, 255, 255);
constexpr sf::Color NEURON_BG_COLOR(0, 0, 100);

constexpr float MIN_NEURON_WIDTH = 6.0f;
constexpr float MAX_NEURON_WIDTH = NEURON_WIDTH;
constexpr float MIN_GAP = 2.0f;

constexpr float FRACTION_ALONG_LINE = 0.8f;

enum class textType {
	UP,
	DOWN,
	NORMAL,
};

static const std::array<sf::Color, 3> color_lookup = {
    FONT_COLOR_UP,
    FONT_COLOR_DOWN,
    FONT_COLOR_NORMAL,
};

class visualLayer : public model::ILayer, public Panel {
  private:
	void clear();
	void display();
	void doRender() override;

	void drawNeurons();
	void doCacheNeurons();
	static sf::Color getNeuronColor(const global::ValueType value);

	virtual textType getTextT(const int layer_i, const int layer_p);
	virtual void renderNeuron(const int index) = 0;

  protected:
	const std::uint32_t WIDTH;
	sf::RenderTexture layerRender;
	std::vector<sf::FloatRect> cacheNeurons;

	static float getScaleFactor(const std::size_t neuron_count);
	static float calculateDistance(const sf::Vector2f pos1, const sf::Vector2f pos2);
	static sf::Angle calculateAngle(const sf::Vector2f pos1, const sf::Vector2f pos2);
	static float calculateGap(const int size, const float scale);

	void drawNeuron(const sf::FloatRect &rect, const double input, const double output);

  public:
	visualLayer(const int _size, const int _prev_size, const std::shared_ptr<StateManager> state_, const std::uint32_t width);
	virtual ~visualLayer() = default;

	model::LayerType getType() const override { return model::LayerType::NONE; }
	sf::Sprite getSprite();
	void set_weights(const model::LayerParameters &Param);
	void setDots(const model::Neurons &newNeurons);
	std::uint32_t getWidth() { return WIDTH; }
};

class visualEmptyLayer : public visualLayer {
  private:
	textType getTextT(const int, const int) override;
	void renderNeuron(const int index) override;

  public:
	visualEmptyLayer(const int _size, const std::shared_ptr<StateManager> state_)
	    : visualLayer(_size, 0, state_, NEURON_WIDTH) {}
	~visualEmptyLayer() = default;
};

struct cacheWeight {
	sf::Angle angle;
	float lineLength;
	sf::Vector2f posLine;
	sf::Vector2f posText;
	sf::Vector2f pos;
};

struct cacheParam {
	std::uint32_t fontSize;
	std::vector<cacheWeight> Weights;

	cacheParam(const int size) : Weights(size) {}
};

class visualParamLayer : public visualLayer {
  private:
	model::LayerParameters grad;
	std::vector<sf::FloatRect> cachePrevNeurons;

	static sf::Color getColorFromTextT(const textType text_type);
	textType getTextT(const int layer_i, const int layer_p) override;
	void drawWeights(const int neuron_i);
	void renderNeuron(const int index) override;
	void doCacheWeights();

  public:
	visualParamLayer(const int _size, const int _prev_size, const std::shared_ptr<StateManager> state_)
	    : visualLayer(
	          _size,
	          _prev_size,
	          state_,
	          calculate_width(state_->config.network_config.hidden_layer_count() + 1)),
	      grad(_size, _prev_size, 0.5),
	      cachePrevNeurons(_prev_size) {
		doCacheWeights();
	}
	~visualParamLayer() = default;

	void updateGrad(const model::LayerParameters &new_grad);
};
} // namespace nn::visualizer

#endif // VISUALL
