#include "panel.hpp"

namespace nn {
namespace Visualizer {
int panel::render() {
	observe();
	if (need_update) {
		do_render();
		need_update = false;
		return true;
	}

	return false;
}
} // namespace Visualizer
} // namespace nn
