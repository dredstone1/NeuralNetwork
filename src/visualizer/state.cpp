#include "state.hpp"

namespace nn {
namespace Visualizer {
void state::toggle(states state_) {
	switch (state_) {
	case states::Pause:
		settings.pause.store(!settings.pause.load());
		break;
	case states::PreciseMode:
		settings.preciseMode.store(!settings.preciseMode.load());
		break;
	case states::AutoPause:
		settings.autoPause.store(!settings.autoPause.load());
		break;
	default:
		break;
	}
}

std::string state::getStateString(states state_) {
	return statesName[(int)state_];
}

states state::getStatefromString(const std::string &state_) {
	for (int i = 0; i < STATES_COUNT; i++) {
		if (!statesName[i].compare(state_)) {
			return (states)i;
		}
	}

	return states::None;
}

bool state::getState(states state_) {
	switch (state_) {
	case states::Pause:
		return settings.pause.load();
		break;
	case states::PreciseMode:
		return settings.preciseMode.load();
		break;
	case states::AutoPause:
		return settings.autoPause.load();
		break;
	default:
		return 0;
		break;
	}
}

void state::setState(states state_, bool stateM) {
	switch (state_) {
	case states::Pause:
		settings.pause.store(stateM);
		break;
	case states::PreciseMode:
		settings.preciseMode.store(stateM);
		break;
	case states::AutoPause:
		settings.autoPause.store(stateM);
		break;
	default:
		break;
	}
}
} // namespace Visualizer
} // namespace nn
