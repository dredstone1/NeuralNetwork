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
	void drawNeurons(const LayerParameters &new_grad);
	void drawNeurons();
	void drawWeights(const int neuron_i, const sf::Vector2f pos, const float prevGap, const std::vector<double> &new_grad);
	void drawWeights(const int neuron_i, const sf::Vector2f pos, const float prevGap);
	static sf::Color getBGcolor(const bool render);
	static float calculateGap(const float size);
	static float calculateDistance(const sf::Vector2f pos1, const sf::Vector2f pos2);
	static float calculateAngle(const sf::Vector2f pos1, const sf::Vector2f pos2);
	static float calculateWIDTH(const int size_a, const bool is_params);
	static textT getTextT(const double change);
	static sf::Color getColorFromTextT(const textT text_type);

  public:
	visualL(const int _size, const int _prev_size, const int size_a);
	visualL(const Layer &other, const int size_a);
	LayerType getType() const override { return LayerType::NONE; }
	sf::Sprite getSprite();
	void renderLayer(const bool render, const LayerParameters &new_grad);
	void renderLayer(const bool render);
	void setDots(const std::vector<double> out, const std::vector<double> net);
	const bool is_params;
	const float WIDTH;
};
} // namespace Visualizer

#endif // VISUALL
