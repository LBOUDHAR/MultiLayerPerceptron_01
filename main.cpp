// main.cpp for Multi Layer Perceptron network: 
// Program created on 21.06.2022 by Lies BOUDHAR
// Modified on 10.08.2022	at 09:35 PM

#include <windows.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include "mlpnetwork.cpp"

using namespace std;

vector<vector<float>> inputs;
string line;

//load data for training from .csv file
void load_data(string& _filename){
  	fstream file(_filename);
	vector<float> v;
  	if (!file.is_open())  {  cerr << "failed to open file\n"; } // check file is open, error message if not
	while (getline(file, line)){
		stringstream sep(line);
		string field;
		while (getline(sep, field, ',')){
			v.push_back(stof(field));
        }
		inputs.push_back(v);
		v.clear();
	}
	file.close();
}


void showTrainData(){
	for (auto row : inputs) {
        for (auto field : row) {
            cout << field << ' ';
        }
        cout << '\n';
    }
/*
for(unsigned int i = 0; i < inputs.size(); i++){
		for (unsigned int j = 0; j < inputs[i].size(); j++) {
			cout << inputs[i][j] << "  ";
		}
		cout << endl;
	}
*/	
	cout << endl;
}


int main(int argc, char *argv[]) {

	string filename = "data_input.csv";

   	cout << "MLP Network by Lies BOUDHAR 06.2022 \n\n";

	load_data(filename);
	
	showTrainData();

	vector <float> outputs = { 1, 0 };
	vector <unsigned int> topology = {4, 3, 2};
	topology.insert(topology.begin(), inputs[0].size());
	MLP_Network network01(topology, inputs[0], outputs);
	network01.showWeightsVals();
	network01.showLayersZVals();
	network01.showLayersAVals();
	network01.showBiasVals();
	network01.forwardPropagat(SEGMOID, SOFTMAX);
	network01.showLayersZVals();
	network01.showLayersAVals();
    cout << endl; cout <<"press ESC to exit ";
	BOOL key = true;
	while (key == true) {
		if (GetKeyState(VK_ESCAPE) & 0x8000){
		key = false;
		}
	}
	return 0;
}