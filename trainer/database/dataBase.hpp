#ifndef DATABASE_HPP
#define DATABASE_HPP

#include <string>
#include <vector>

using namespace std;

#define MAX_BOARDS_COUNT 10000
#define AMOUNT_OF_CELLS 9

struct TrainBoard {
	vector<double> board;
	int best_next_move;
};

typedef vector<TrainBoard> DataBaseVector;

using namespace std;

class DataBase {
  private:
	const string file_name;
	int load();
	TrainBoard read_line(const string &line);
	static void analyze_board(vector<double> &board, const char *board_str);
	DataBaseVector train_boards[AMOUNT_OF_CELLS];
	void insert_board(const TrainBoard &board);
	int static is_X_Turn(const vector<double> &board);
	const vector<double> &get_board(const int index) const;
	void shrink_to_fit();

  public:
	DataBase(const string &_file_name);
	int get_train_boards_count() const;
	vector<TrainBoard> get_Batch(const int batch_size);
	~DataBase() = default;
};

#endif // DATABASE_HPP
