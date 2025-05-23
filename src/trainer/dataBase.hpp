#ifndef DATABASE_HPP
#define DATABASE_HPP

#include "AiModel.hpp"
#include <string>
#include <vector>

using namespace std;

typedef struct TrainSample {
	prediction _prediction;
	vector<double> input;
	TrainSample(prediction _pre, const int sampleInputSize) : _prediction(_pre), input(sampleInputSize, 0) {}
	TrainSample() : _prediction({0, 0}), input(0) {}
} TrainSample;

typedef struct Samples {
	const int sInputSize;
	vector<TrainSample> samples;
	int size() const { return samples.size(); }
	void add(TrainSample sample);
	Samples(const int sampleInputSize, const int _size) : sInputSize(sampleInputSize), samples(_size) {}
} Samples;

typedef struct Batch {
	vector<TrainSample *> samples;
	const int sampleSize;
	int size() const { return samples.size(); }
	Batch(const int length, const int _sampleSize) : sampleSize(_sampleSize) { samples.reserve(length); }
	void add(TrainSample *sample) { samples.push_back(sample); }
} Batch;

class DataBase {
  private:
	const string file_name;
	void getDataBaseStatus(const string &line);
	TrainSample read_line(const string &line);
	Samples *samples;
	int load();

  public:
	DataBase(const string &_file_name);
	int DataBaseLength() const { return samples->size(); }
	Batch get_Batch(const int batch_size);
	~DataBase() = default;
};

#endif // DATABASE_HPP
