#ifndef GRAPH
#define GRAPH

#include "panel.hpp"
#include <SFML/Graphics.hpp>

namespace Visualizer {
constexpr std::uint32_t GRAPH_WIDTH = 470;
constexpr std::uint32_t GRAPH_HEIGHT = 315;
constexpr std::uint32_t GRAPH_UI_WIDTH = 500;

constexpr int GRAPH_TEXT_FONT = 30;
constexpr std::uint32_t GRAPH_RESOLUTION = 100;
constexpr std::uint32_t DATA_GAP_WIDTH = GRAPH_WIDTH / GRAPH_RESOLUTION;
constexpr float GRAPH_HEIGHT_ALPHA_DEFAULT = 100.f;
constexpr int VERTICAL_NUMBERS_COUNT = 7;

constexpr sf::Color GRAPH_LINE_COLOR(0, 0, 0);
constexpr sf::Color GRAPH_VERTICAL_NUMBER_COLOR(0, 0, 0);
constexpr sf::Color GRAPH_BG = PANELS_BG;

class GraphUI : public panel {
  private:
	std::array<double, GRAPH_RESOLUTION> data;
	sf::RenderTexture VRender;
	sf::RenderTexture Vgraph;
	void display();
	void clear();
	void renderGraph();
	void render_numbers();
	void render_vertical_numbers();
	void do_render() override;
	inline int data_gaps();
	int newDataPlace(const int index);
	void renderDot(const int index);
	int get_highest();
	double graph_alpha;
	sf::Vector2f getPosition(const int index);
	float get_hight(const int index);
	inline std::uint32_t resolution();
	inline float data_gap_width();

  public:
	GraphUI(const std::shared_ptr<state> vstate_);
	sf::Sprite getSprite();
	void add_data(const float new_data, const int index);
};
} // namespace Visualizer

#endif // GRAPH
