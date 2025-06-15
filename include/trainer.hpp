#ifndef TRAINER
#define TRAINER

#include "../src/trainer/backPropagation.hpp"

namespace nn::training {
constexpr int BAR_WIDTH = 100;
constexpr int SECONDS_IN_MINUTE = 60;

class Trainer {
  private:
	model::TrainingConfig &config;
	DataBase dataBase;
	AiModel &model;
	BackPropagation backPropagation;
	int last_progress;
	learningRateParams learningRate;
	void print_progress_bar(const int current, const int total);

  public:
	Trainer(AiModel &_model)
	    : config(_model.getConfig().config_data.training_config),
	      dataBase(config),
	      model(_model),
	      backPropagation(_model, learningRate),
	      last_progress(-1) {}
	void train();
	~Trainer() = default;
};
} // namespace nn::training

#endif // TRAINER
