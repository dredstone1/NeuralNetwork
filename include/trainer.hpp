#ifndef TRAINER_HPP
#define TRAINER_HPP

#include "../src/trainer/backPropagation.hpp"
#include "../src/trainer/dataBase.hpp"
#include "AiModel.hpp"

#define GRAPH_RESOLUTION 100
#define BAR_WIDTH 100
#define SECONDS_IN_MINUTE 60

class Trainer {
  private:
	string file_name;
	DataBase dataBase;
	AiModel *model;
	BackPropagation backPropagation;
	int batch_size;
	int batch_count;
	void print_progress_bar(int current, int total);
	int last_progress;
	double learning_rate;

  public:
	Trainer(string _file_name, AiModel *_model, int _batch_size, int _batch_count, double _learning_rate);
	void train();
	~Trainer() = default;
};

#endif // TRAINER_HPP
