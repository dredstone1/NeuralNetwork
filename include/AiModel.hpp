#ifndef AIMODEL
#define AIMODEL

#include "../src/model/config.hpp"
#include "../src/model/model.hpp"
#include <Globals.hpp>
#include <string>

namespace nn {
struct prediction {
	const int index;
	const Global::ValueType value;
	prediction(const size_t index_, const Global::ValueType value_)
	    : index(index_),
	      value(value_) {}
	prediction(const prediction &other)
	    : index(other.index),
	      value(other.value) {}
};

class AiModel {
  private:
	std::unique_ptr<model> _model;
	Config config;
	friend class BackPropagation;
	friend class Trainer;

  public:
	AiModel(const std::string &config_file, const bool use_visual = false);
	void run_model(const std::vector<Global::ValueType> &input);
	prediction getPrediction();
	Config &getConfig() { return config; }
	~AiModel() = default;
};
} // namespace nn

#endif // AIMODEL
