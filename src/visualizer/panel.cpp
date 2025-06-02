#include "panel.hpp"

namespace Visualizer {
int panel::render() {
	if (need_update) {
		do_render();
		need_update = false;
		return true;
	}

	return false;
}
} // namespace Visualizer
