#ifndef FNNVISUALNETWORK
#define FNNVISUALNETWORK

#include "IvisualNetwork.hpp"

namespace nn::visualizer {
class FnnVisualier : public IVisualNetwork {
  private:
	void renderNetwork() override;
	void createNetwork() override;

  public:
	FnnVisualier(
	    const std::shared_ptr<StateManager> state_,
	    const std::uint32_t width);
	~FnnVisualier() = default;
};
} // namespace nn::visualizer

#endif // FNNVISUALNETWORK
