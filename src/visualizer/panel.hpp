#ifndef PANEL
#define PANEL

#include "state.hpp"
#include <SFML/Graphics/Color.hpp>

namespace nn {
namespace Visualizer {
constexpr sf::Color PANELS_BG(255, 255, 255);

class panel {
  private:
	virtual void do_render() = 0;
	bool need_update{true};
	virtual void observe() {}

  protected:
	std::shared_ptr<state> vstate;

  public:
	panel(const std::shared_ptr<state> vstate_)
	    : vstate(vstate_) {}
	int render();
	virtual ~panel() = default;
	bool updateStatus() const { return need_update; }
	void set_update() { need_update = true; }
};
} // namespace Visualizer
} // namespace nn

#endif // PANEL
