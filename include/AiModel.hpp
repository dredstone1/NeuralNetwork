#ifndef AIMODEL
#define AIMODEL

#include "../src/model/model.hpp"
#include <string>

namespace nn {
class AiModel {
  private:
	std::unique_ptr<model::Model> model;
	model::Config config;

  public:
	AiModel(const std::string &config_file);
	~AiModel() = default;

	void runModel(const global::ParamMetrix &input);
	void train();

	global::Prediction getPrediction();
	model::Config &getConfig() { return config; }
};
} // namespace nn

#endif // AIMODEL
