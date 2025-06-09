#ifndef GRAPH
#define GRAPH

#include "panel.hpp"
#include <SFML/Graphics.hpp>

namespace nn::visualizer {
constexpr std::uint32_t GRAPH_WIDTH = 470;
constexpr std::uint32_t GRAPH_HEIGHT = 315;
constexpr std::uint32_t GRAPH_UI_WIDTH = 500;

constexpr int GRAPH_TEXT_FONT = 30;
constexpr std::uint32_t GRAPH_RESOLUTION = 100;
constexpr std::uint32_t DATA_GAP_WIDTH = GRAPH_WIDTH / GRAPH_RESOLUTION;
constexpr float GRAPH_HEIGHT_ALPHA_DEFAULT = GRAPH_HEIGHT;
constexpr int VERTICAL_NUMBERS_COUNT = 10;

constexpr sf::Color GRAPH_LINE_COLOR(0, 0, 0);
constexpr sf::Color GRAPH_VERTICAL_NUMBER_COLOR(0, 0, 0);
constexpr sf::Color GRAPH_BG = PANELS_BG;

class GraphUI : public Panel {
  private:
	std::array<global::ValueType, GRAPH_RESOLUTION> data;
	sf::RenderTexture VRender;
	sf::RenderTexture Vgraph;
	double graphAlpha;

	void display();
	void clear();
	void renderGraph();
	void render_numbers();
	void renderVerticalNumbers();
	void doRender() override;
	inline int data_gaps();
	int newDataPlace(const int index);
	void renderDot(const int index);
	int get_highest();
	sf::Vector2f getPosition(const int index);
	float get_height(const int index);
	float get_height(const float index);
	float getValueFromHeight(const float height);
	inline std::uint32_t resolution();
	inline float data_gap_width();
	void renderHorizontalLine(const float value);

  public:
	GraphUI(const std::shared_ptr<StateManager> vstate_);
	sf::Sprite getSprite();
	void add_data(const global::ValueType new_data, const int index);
	~GraphUI() = default;
};
} // namespace nn::visualizer

#endif // GRAPH
