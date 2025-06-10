#ifndef STATE
#define STATE

#include "../model/config.hpp"
#include <atomic>

namespace nn::visualizer {
constexpr int STATES_COUNT = 4;

enum class SettingType {
	Pause,
	PreciseMode,
	AutoPause,
	ExitTraining,
	None,
};

const std::array<std::string_view, STATES_COUNT> statesName = {
    "pause",
    "precise mode",
    "auto pause",
    "exit training",
};

enum class NnMode {
	Forword,
	Backward,
	None,
};

enum class AlgorithmMode {
	Normal,
	Training,
};

struct Settings {
	std::atomic<bool> pause{true};
	std::atomic<bool> preciseMode{true};
	std::atomic<bool> autoPause{true};
	std::atomic<bool> exitTraining{false};
};

class StateManager {
  public:
	Settings settings;
	int currentBatch{0};
	const model::ConfigData config;
	std::atomic<bool> updateMode{false};
	std::atomic<NnMode> nnMode{NnMode::Forword};
	std::atomic<AlgorithmMode> algorithmMode{AlgorithmMode::Normal};

	StateManager(const model::ConfigData config)
	    : config(config) {}
	void toggle(const SettingType state);
	void toggle(const std::string state) { toggle(getStatefromString(state)); }
	std::string_view getStateString(const SettingType state);
	SettingType getStatefromString(const std::string &state);
	bool getState(const SettingType state);
	void setState(const SettingType state, const bool stateMode);
	void setState(const std::string &state, const bool stateMode) { setState(getStatefromString(state), stateMode); }
	~StateManager() = default;
};
} // namespace nn::visualizer

#endif // STATE
