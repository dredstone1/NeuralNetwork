#ifndef AIMODEL
#define AIMODEL

#include "../src/model/model.hpp"
#include <Globals.hpp>
#include <string>

namespace nn {

struct Prediction {
	const int index;
	const global::ValueType value;
	Prediction(const size_t index_, const global::ValueType value_)
	    : index(index_),
	      value(value_) {}
	Prediction(const Prediction &other)
	    : index(other.index),
	      value(other.value) {}
};

class AiModel {
  private:
	std::unique_ptr<model::Model> model;
	model::Config config;

	friend class training::BackPropagation;
	friend class training::Trainer;

  public:
	AiModel(const std::string &config_file, const bool use_visual = false);
	void runModel(const global::ParamMetrix &input);
	Prediction getPrediction();
	model::Config &getConfig() { return config; }
	~AiModel() = default;
};
} // namespace nn

#endif // AIMODEL
