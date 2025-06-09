#include "panel.hpp"

namespace nn::visualizer {
int Panel::render() {
	observe();
	if (need_update) {
		doRender();
		need_update = false;
		return true;
	}

	return false;
}
} // namespace nn::visualizer
