#ifndef TRAINER
#define TRAINER

#include "../src/trainer/backPropagation.hpp"
#include "../src/trainer/dataBase.hpp"
#include <AiModel.hpp>

constexpr int BAR_WIDTH = 100;
constexpr int SECONDS_IN_MINUTE = 60;

namespace nn {
class Trainer {
  private:
	TrainingConfig &config;
	DataBase dataBase;
	AiModel &model;
	LearningRate lr;
	BackPropagation backPropagation;
	void print_progress_bar(const int current, const int total);
	int last_progress;

  public:
	Trainer(AiModel &_model)
	    : config(_model.getConfig().config_data.training_config),
	      dataBase(config),
	      model(_model),
	      lr(config),
	      backPropagation(_model, lr),
	      last_progress(-1) {}
	void train();
	~Trainer() = default;
};
} // namespace nn

#endif // TRAINER
