#ifndef STATE
#define STATE

#include <atomic>
#include <string>

namespace Visualizer {
#define STATES_COUNT 3

enum class states {
	Pause,
	PreciseMode,
	AutoPause,
	None,
};

const std::string statesName[] = {
    "pause",
    "precise mode",
    "auto pause",
};

enum class NNmode {
	Forword,
	Backward,
	None,
};

const std::string NNmodeName[] = {
    "Forword",
    "Backward",
};

typedef struct state {
	std::atomic<bool> pause{false};
	std::atomic<bool> preciseMode{true};
	std::atomic<bool> autoPause{true};
	std::atomic<bool> update_mode{false};
	std::atomic<NNmode> nnMode{NNmode::Forword};
	void toggle(const states state_);
	void toggle(const std::string state_) { toggle(getStatefromString(state_)); }
	std::string getStateString(const states state_);
	states getStatefromString(const std::string &_state);
	bool getState(const states state_);
	void setState(const states state_, const bool stateM);
	void setState(const std::string &state_, const bool stateM) { setState(getStatefromString(state_), stateM); }
    ~state() = default;
} state;
} // namespace Visualizer

#endif // STATE
