#ifndef VISUALL
#define VISUALL

#include "../model/LayerParameters.hpp"
#include "../model/Layers/layer.hpp"
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/System/Vector2.hpp>
#include <vector>

namespace Visualizer {
#define NN_HEIGHT 770
#define NEURON_RADIUS 20.f
#define NN_WIDTH 1055

enum class textT {
	UP,
	DOWN,
	NORMAL
};

class visualL : public Layer {
  private:
	sf::RenderTexture layerRender;
	void createLayerVisual();
	void clear(const bool render);
	void display();
	void drawNeuron(const double input, const double output, sf::Vector2f pos);
	void drawNeurons();
	void drawWeights(const int neuron_i, const sf::Vector2f pos, const float prevGap);
	static sf::Color getBGcolor(const bool render);
	static float calculateGap(const float size);
	static float calculateDistance(const sf::Vector2f pos1, const sf::Vector2f pos2);
	static float calculateAngle(const sf::Vector2f pos1, const sf::Vector2f pos2);
	static float calculateWIDTH(const int size_a, const bool is_params);
	virtual textT getTextT(const int layer_i, const int layer_p);
	static sf::Color getColorFromTextT(const textT text_type);

  public:
	visualL(const int _size, const int _prev_size, const int size_a);
	visualL(const Layer &other, const int size_a);
	LayerType getType() const override { return LayerType::NONE; }
	sf::Sprite getSprite();
	void renderLayer(const bool render);
	void setDots(const std::vector<double> out, const std::vector<double> net);
	const bool is_params;
	const float WIDTH;
	virtual ~visualL() = default;
};

class VEmptyLayer : public visualL {
  private:
	textT getTextT(const int, const int) override;

  public:
	VEmptyLayer(const int _size, const int _prev_size, const int size_a) : visualL(_size, _prev_size, size_a) {}
	VEmptyLayer(const Layer &other, const int size_a) : visualL(other, size_a) {}
	~VEmptyLayer() = default;
};

class VParamLayer : public visualL {
  private:
	LayerParameters grad;
	textT getTextT(const int layer_i, const int layer_p) override;

  public:
	VParamLayer(const int _size, const int _prev_size, const int size_a) : visualL(_size, _prev_size, size_a), grad(_size, _prev_size, 0.5) {}
	VParamLayer(const Layer &other, const int size_a) : visualL(other, size_a), grad(other.getSize(), other.getPrevSize(), 0.5) {}
	void updateGrad(const LayerParameters &new_grad);
	~VParamLayer() = default;
};
} // namespace Visualizer

#endif // VISUALL
