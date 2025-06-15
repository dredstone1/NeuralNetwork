#ifndef AIMODEL
#define AIMODEL

#include "../src/model/model.hpp"
#include <string>

namespace nn {
struct Prediction {
	const int index;
	const global::ValueType value;
	Prediction() : index(0), value(0) {}
	Prediction(const int index_, const global::ValueType value_)
	    : index(index_),
	      value(value_) {}
};

class AiModel {
  private:
	std::unique_ptr<model::Model> model;
	model::Config config;

	friend class training::BackPropagation;
	friend class training::Trainer;

  public:
	AiModel(const std::string &config_file);
	~AiModel() = default;

	void runModel(const global::ParamMetrix &input);
	Prediction getPrediction();
	model::Config &getConfig() { return config; }
};
} // namespace nn

#endif // AIMODEL
