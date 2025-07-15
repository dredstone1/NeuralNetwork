#ifndef GRAPH_CORE_HPP
#define GRAPH_CORE_HPP

#include "panel.hpp"
#include <SFML/Graphics.hpp>

namespace nn::visualizer {
constexpr std::uint32_t GRAPH_WIDTH = 440;
constexpr std::uint32_t GRAPH_HEIGHT = 315;
constexpr std::uint32_t GRAPH_RESOLUTION = 100;
constexpr int VERTICAL_NUMBERS_COUNT = 10;

constexpr std::uint32_t GRAPH_UI_WIDTH = 500;

constexpr int GRAPH_TEXT_FONT = 30;
constexpr std::uint32_t DATA_GAP_WIDTH = GRAPH_WIDTH / GRAPH_RESOLUTION;
constexpr float GRAPH_HEIGHT_ALPHA_DEFAULT = GRAPH_HEIGHT;

constexpr sf::Color GRAPH_LINE_COLOR_LOST(0, 0, 0);
constexpr sf::Color GRAPH_LINE_COLOR_EVALUATE(255, 0, 255);
constexpr sf::Color GRAPH_HORIZONTAL_LINE_COLOR(0, 0, 255);
constexpr sf::Color GRAPH_BG = PANELS_BG;

class Graph {
  private:
	std::array<global::ValueType, GRAPH_RESOLUTION> data{0};
	double graphAlpha;
	std::uint32_t resolution;

	float getHeight(const float value) const;
	sf::Vector2f getPosition(const int index) const;
	void renderDot(
	    const int index,
	    sf::RenderTarget &target,
	    const sf::Vector2f &position,
	    const sf::Color &color);
	float dataGapWidth() const;

  public:
	Graph(std::uint32_t resolution = GRAPH_RESOLUTION, float alpha = GRAPH_HEIGHT);
	~Graph() = default;

	void drawTo(sf::RenderTarget &rarget, const sf::Vector2f position = {0.f, 0.f}, const sf::Color &color = sf::Color::Black);
	void addData(const global::ValueType new_data, const int index, const int batchCount);
	void setAlpha(const float alpha);
	float getAlpha() const;
};

class GraphUIPanel : public Panel {
  private:
	sf::RenderTexture VRender;
	Graph graphLost;
	Graph graphEvaluate;

	void renderVerticalNumbers();
	void renderHorizontalLine(float value);
	float getValueFromHeightE(float height);
	float getValueFromHeightL(float height);
	void display();
	void clear();
	void doRender() override;

  public:
	GraphUIPanel(std::shared_ptr<StateManager> vstate_);
	~GraphUIPanel() = default;

	sf::Sprite getSprite();
	void addData(
	    const global::ValueType newDataEvaluate,
	    const global::ValueType newDataLost,
	    int index);
};

} // namespace nn::visualizer

#endif // GRAPH_CORE_HPP
