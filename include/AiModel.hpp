#ifndef AIMODEL
#define AIMODEL

#include "../src/model/model.hpp"
#include <string>

namespace nn {
class AiModel {
  private:
	std::unique_ptr<model::Model> model;

  public:
	AiModel(const std::string &config_file);
	~AiModel() = default;

	void runModel(const global::ParamMetrix &input);
	void train();

	global::Prediction getPrediction();
};
} // namespace nn

#endif // AIMODEL
