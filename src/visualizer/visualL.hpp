#ifndef VISUALL
#define VISUALL

#include "../model/Layers/layer.hpp"
#include "panel.hpp"
#include <SFML/Graphics.hpp>

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

enum class textT {
	UP,
	DOWN,
	NORMAL,
};

class visualL : public model::Layer, public Panel {
  private:
	void clear();
	void display();
	void drawNeurons();
	virtual textT getTextT(const int layer_i, const int layer_p);
	void doRender() override;
	virtual void renderNeuron(const int index, const float gap, const float prevGap, const float scale) = 0;
	float getScaleFactor(std::size_t neuron_count) const;

  protected:
	sf::RenderTexture layerRender;
	static std::uint32_t calculateGap(const float size);
	static float calculateDistance(const sf::Vector2f pos1, const sf::Vector2f pos2);
	static float calculateAngle(const sf::Vector2f pos1, const sf::Vector2f pos2);
	void drawNeuron(const double input, const double output, const sf::Vector2f pos, float scale);

  public:
	visualL(const int _size, const int _prev_size, const std::shared_ptr<StateManager> state_, const std::uint32_t width);
	visualL(const Layer &other, const std::shared_ptr<StateManager> state_, const std::uint32_t width);
	model::LayerType getType() const override { return model::LayerType::NONE; }
	sf::Sprite getSprite();
	void set_weights(const model::LayerParameters &Param);
	void setDots(const model::Neurons &newNeurons);
	const std::uint32_t WIDTH;
	virtual ~visualL() = default;
};

class VEmptyLayer : public visualL {
  private:
	textT getTextT(const int, const int) override;
	void renderNeuron(const int index, const float gap, const float, const float scale) override;

  public:
	VEmptyLayer(const int _size, const std::shared_ptr<StateManager> state_)
	    : visualL(_size, 0, state_, NEURON_WIDTH) {}
	VEmptyLayer(const Layer &other, const std::shared_ptr<StateManager> state_)
	    : visualL(other, state_, NEURON_WIDTH) {}
	~VEmptyLayer() = default;
};

class VParamLayer : public visualL {
  private:
	model::LayerParameters grad;
	static sf::Color getColorFromTextT(const textT text_type);
	textT getTextT(const int layer_i, const int layer_p) override;
	void drawWeights(const int neuron_i, const sf::Vector2f pos, const float prevGap, float scale);
	void renderNeuron(const int index, const float gap, const float prevGap, const float scale) override;

  public:
	VParamLayer(const int _size, const int _prev_size, const std::shared_ptr<StateManager> state_)
	    : visualL(_size, _prev_size, state_, calculate_width(state_->config.network_config.hidden_layer_count() + 1)),
	      grad(_size, _prev_size, 0.5) {}
	VParamLayer(const Layer &other, const std::shared_ptr<StateManager> state_)
	    : visualL(other, state_, calculate_width(state_->config.network_config.hidden_layer_count() + 1)),
	      grad(other.getSize(), other.getPrevSize(), 0.5) {}
	void updateGrad(const model::LayerParameters &new_grad);
	~VParamLayer() = default;
};
} // namespace nn::visualizer

#endif // VISUALL
