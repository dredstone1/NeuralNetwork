#include "dataBase.hpp"
#include "../../include/AiPlayer/AiPlayer.hpp"
#include "../../utils.hpp"
#include <climits>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string.h>
#include <string>
#include <vector>

using namespace std;

DataBase::DataBase(const string &_file_name) : file_name(_file_name) {
	load();
}

void DataBase::analyze_board(vector<double> &board, const char *board_str) {
	for (size_t i = 0; i < 9; i++) {
		switch (board_str[i]) {
		case 'X':
			board[i] = CELL_VALUE_X;
			break;
		case 'O':
			board[i] = CELL_VALUE_O;
			break;
		case ' ':
			break;
		case ',':
			return;
		default:
			break;
		}
	}
	return;
}

TrainBoard DataBase::read_line(const string &line) {
	vector<double> board_values(9, CELL_VALUE_EMPTY);
	int best_next_move = 0;

	char cstr[50];
	strcpy(cstr, line.c_str());
	char *token = strtok(cstr, ":");

	while (token != NULL) {
		if (strcmp(token, "Board") == 0) {
			char *board_str = strtok(NULL, ",");
			analyze_board(board_values, board_str);
		} else if (strcmp(token, " Best_Next_Move") == 0) {
			char *move_str = strtok(NULL, "");
			if (strcmp(move_str, " Game Over") == 0) {
				return {board_values, -1};
			}

			best_next_move = stoi(move_str);
		}
		token = strtok(NULL, ":");
	}

	return {board_values, best_next_move};
}

int DataBase::is_X_Turn(const vector<double> &board) {
	int x_count = 0;
	int o_count = 0;

	for (size_t i = 0; i < 9; i++) {
		if (board[i] == CELL_VALUE_X) {
			x_count++;
		} else if (board[i] == CELL_VALUE_O) {
			o_count++;
		}
	}
	// X's turn if x_count == o_count, return 0 if it's X's turn
	return x_count == o_count;
}

void DataBase::insert_board(const TrainBoard &board) {
	train_boards[board.best_next_move].push_back(board);
}

int DataBase::get_train_boards_count() const {
	int count = 0;
	for (int i = 0; i < AMOUNT_OF_CELLS; i++) {
		count += train_boards[i].size();
	}
	return count;
}

void DataBase::shrink_to_fit() {
	for (int i = 0; i < AMOUNT_OF_CELLS; i++) {
		train_boards[i].shrink_to_fit();
	}
}

int DataBase::load() {
	ifstream file(file_name + ".txt");
	if (!file.is_open()) {
		cout << "File not found: " << file_name << endl;
		return 1;
	}
	string line;
	getline(file, line);
	int count = atoi(strchr(line.c_str(), ':') + 1);
	count = MIN(count, MAX_BOARDS_COUNT);
	getline(file, line);

	while (count >= 0 && getline(file, line)) {
		TrainBoard board = read_line(line);
		if (board.best_next_move == -1) {
			continue;
		}
		count--;

		insert_board(board);
	}

	shrink_to_fit();
	cout << "Loaded " << get_train_boards_count() << " boards." << endl;
	file.close();

	return 0;
}

vector<TrainBoard> DataBase::get_Batch(const int batch_size) {
	vector<TrainBoard> batch;
	for (int i = 0; i < batch_size; i++) {
		int index = rand() % AMOUNT_OF_CELLS;
		if (train_boards[index].size() == 0) {
			i--;
			continue;
		}
		int board_index = rand() % train_boards[index].size();
		batch.push_back(train_boards[index][board_index]);
	}
	return batch;
}
