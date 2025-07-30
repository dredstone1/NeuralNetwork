#include "panel.hpp"

namespace nn::visualizer {
int Panel::render() {
	observe();
	if (updateStatus()) {
		doRender();
		need_update = false;
		return true;
	}

	return false;
}

void Panel::setUpdate(const bool) {
	need_update = true;
}
} // namespace nn::visualizer
