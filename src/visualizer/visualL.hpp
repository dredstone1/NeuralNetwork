#ifndef VISUALL
#define VISUALL

#include "../model/Layers/layer.hpp"
#include "panel.hpp"
#include <SFML/Graphics.hpp>
#include <memory>
#include <vector>

namespace Visualizer {
constexpr std::uint32_t NN_HEIGHT = 770u;
constexpr std::uint32_t NN_WIDTH = 1055u;

constexpr std::uint32_t NEURON_RADIUS = 20;
constexpr std::uint32_t NEURON_WIDTH = NEURON_RADIUS * 2;

constexpr std::uint32_t calculate_width(const int layer_amount) {
	return (NN_WIDTH - NEURON_WIDTH) / layer_amount;
}

constexpr sf::Color NORMAL_BG_LAYER(255, 255, 255);
constexpr sf::Color ACTIVE_BG_LAYER(187, 187, 187);

enum class textT {
	UP,
	DOWN,
	NORMAL,
};

class visualL : public Layer, public panel {
  private:
	void clear();
	void display();
	void drawNeurons();
	virtual textT getTextT(const int layer_i, const int layer_p);
	void do_render() override;
	virtual void renderNeuron(const int index, const float gap, const float prevGap) = 0;

  protected:
	sf::RenderTexture layerRender;
	static std::uint32_t calculateGap(const float size);
	static float calculateDistance(const sf::Vector2f pos1, const sf::Vector2f pos2);
	static float calculateAngle(const sf::Vector2f pos1, const sf::Vector2f pos2);
	void drawNeuron(const double input, const double output, sf::Vector2f pos);

  public:
	visualL(const int _size, const int _prev_size, const std::shared_ptr<state> state_, const std::uint32_t width);
	visualL(const Layer &other, const std::shared_ptr<state> state_, const std::uint32_t width);
	LayerType getType() const override { return LayerType::NONE; }
	sf::Sprite getSprite();
	void set_weights(const LayerParameters &Param);
	void setDots(const std::vector<double> &out, const std::vector<double> &net);
	const std::uint32_t WIDTH;
	virtual ~visualL() = default;
};

class VEmptyLayer : public visualL {
  private:
	textT getTextT(const int, const int) override;
	void renderNeuron(const int index, const float gap, const float) override;

  public:
	VEmptyLayer(const int _size, const int _prev_size, const std::shared_ptr<state> state_)
	    : visualL(_size, _prev_size, state_, NEURON_WIDTH) {}
	VEmptyLayer(const Layer &other, const std::shared_ptr<state> state_)
	    : visualL(other, state_, NEURON_WIDTH) {}
	~VEmptyLayer() = default;
};

class VParamLayer : public visualL {
  private:
	LayerParameters grad;
	static sf::Color getColorFromTextT(const textT text_type);
	textT getTextT(const int layer_i, const int layer_p) override;
	void drawWeights(const int neuron_i, const sf::Vector2f pos, const float prevGap);
	void renderNeuron(const int index, const float gap, const float prevGap) override;

  public:
	VParamLayer(const int _size, const int _prev_size, const std::shared_ptr<state> state_)
	    : visualL(_size, _prev_size, state_, calculate_width(state_->config.network_config.hidden_layer_count() + 1)),
	      grad(_size, _prev_size, 0.5) {}
	VParamLayer(const Layer &other, const std::shared_ptr<state> state_)
	    : visualL(other, state_, calculate_width(state_->config.network_config.hidden_layer_count() + 1)),
	      grad(other.getSize(), other.getPrevSize(), 0.5) {}
	void updateGrad(const LayerParameters &new_grad);
	~VParamLayer() = default;
};
} // namespace Visualizer

#endif // VISUALL
