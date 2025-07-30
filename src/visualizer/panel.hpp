#ifndef PANEL
#define PANEL

#include "state.hpp"
#include <SFML/Graphics/Color.hpp>

namespace nn::visualizer {
constexpr sf::Color PANELS_BG(255, 255, 255);

class Panel {
  private:
	virtual void doRender() = 0;
	bool need_update{true};
	virtual void observe() {}

  protected:
	std::shared_ptr<StateManager> vstate;

  public:
	Panel(const std::shared_ptr<StateManager> vstate_)
	    : vstate(vstate_) {}
	virtual ~Panel() = default;

	int render();
	virtual bool updateStatus() const { return need_update; }
	void setUpdate(const bool wait = false);
};
} // namespace nn::visualizer

#endif // PANEL
