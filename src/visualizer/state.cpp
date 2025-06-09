#include "state.hpp"

namespace nn::visualizer {
void StateManager::toggle(SettingType state_) {
	switch (state_) {
	case SettingType::Pause:
		settings.pause.store(!settings.pause.load());
		break;
	case SettingType::PreciseMode:
		settings.preciseMode.store(!settings.preciseMode.load());
		break;
	case SettingType::AutoPause:
		settings.autoPause.store(!settings.autoPause.load());
		break;
	case SettingType::ExitTraining:
		settings.exitTraining.store(!settings.exitTraining.load());
		break;
	default:
		break;
	}
}

std::string_view StateManager::getStateString(SettingType state_) {
	return statesName[(int)state_];
}

SettingType StateManager::getStatefromString(const std::string &state_) {
	for (int i = 0; i < STATES_COUNT; i++) {
		if (!statesName[i].compare(state_)) {
			return (SettingType)i;
		}
	}

	return SettingType::None;
}

bool StateManager::getState(SettingType state_) {
	switch (state_) {
	case SettingType::Pause:
		return settings.pause.load();
		break;
	case SettingType::PreciseMode:
		return settings.preciseMode.load();
		break;
	case SettingType::AutoPause:
		return settings.autoPause.load();
		break;
	case SettingType::ExitTraining:
		return settings.exitTraining.load();
		break;
	default:
		return 0;
		break;
	}
}

void StateManager::setState(SettingType state_, bool stateM) {
	switch (state_) {
	case SettingType::Pause:
		settings.pause.store(stateM);
		break;
	case SettingType::PreciseMode:
		settings.preciseMode.store(stateM);
		break;
	case SettingType::AutoPause:
		settings.autoPause.store(stateM);
		break;
	case SettingType::ExitTraining:
		settings.exitTraining.store(stateM);
		break;
	default:
		break;
	}
}
} // namespace nn
