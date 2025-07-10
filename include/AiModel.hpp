#ifndef AIMODEL
#define AIMODEL

#include "../src/model/model.hpp"
#include <string>

namespace nn {
class AiModel {
  private:
	std::unique_ptr<model::Model> model;

  public:
	AiModel(const std::string &configNile);
	~AiModel() = default;

	void runModel(const global::ParamMetrix &input);
	void train();

    void save(const std::string &file);
    void load(const std::string &file);

	global::FinalPrediction getPrediction();
};
} // namespace nn

#endif // AIMODEL
