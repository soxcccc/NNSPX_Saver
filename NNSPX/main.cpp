#include <iostream>
#include <C:/eigen3/Eigen/Dense>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <chrono>
#include <map>
#include <tuple>

#define stock_rows 465
#define stock_cols 487

using namespace std;

struct result { double mse; Eigen::MatrixXd pred; };

double relu(double x)
{
	return max(0.0, x);
}

double Drelu(double x)
{
	if (x > 0.0)
		return 1.0;
	else if (x < 0.0)
		return 0.0;
	else
		return 0.5;
}

double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

double Dsigmoid(double x)
{
	return sigmoid(x)*(1 - sigmoid(x));
}

class DenseLayer
{
public:
	enum Activation { Sigmoid, ReLu, None };
	enum Optimizer { GD, CGD };//Ludi Wang
	DenseLayer(int _backunits_len, int _units_len, string name, double _learning_rate, bool _is_input_layer, Activation t);
	void Initializer();
	Eigen::MatrixXd ForwardPropagation(const Eigen::MatrixXd &_x_data);
	Eigen::MatrixXd cal_gradient();
	Eigen::MatrixXd BackwardPropagation(const Eigen::MatrixXd & gradient);
	int getbackunits() { return backunits_len; };
	int getunits() { return units_len; };
	Eigen::MatrixXd getweights() { return weight; }
	Eigen::MatrixXd getbias() { return bias; }
	string getname() { return layer_name; }
	void setinputlayer() { is_input_layer = true; };
private:
	Optimizer o;
	Activation act_func;
	int backunits_len; int units_len;
	bool is_input_layer;
	double learning_rate;
	string layer_name;
	Eigen::MatrixXd output;
	Eigen::MatrixXd wx_plus_b;
	Eigen::MatrixXd bias;
	Eigen::MatrixXd weight;
	Eigen::MatrixXd x_data;
	Eigen::MatrixXd gradient_to_prop;
	Eigen::MatrixXd gradient_weight;
	Eigen::MatrixXd gradient_b;
};


DenseLayer::DenseLayer(int _backunits_len, int _units_len, string name, double _learning_rate = 0.03, bool _is_input_layer = false, Activation t = DenseLayer::Sigmoid) :
	output(1, _units_len), wx_plus_b(1, _units_len), bias(1, _units_len), weight(_backunits_len, _units_len), x_data(1, _backunits_len), gradient_to_prop(1, _backunits_len),
	gradient_weight(_backunits_len, _units_len), gradient_b(1, _units_len), layer_name(name)
{
	is_input_layer = _is_input_layer;
	learning_rate = _learning_rate;
	backunits_len = _backunits_len;
	units_len = _units_len;
	act_func = t;

	cout << "Construct a layer " << backunits_len << " to " << units_len << "!" << endl;
}


void DenseLayer::Initializer()
{
	/*weight = Eigen::MatrixXd::Random(backunits_len, units_len);
	bias = Eigen::MatrixXd::Random(1, units_len);*/

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine gen;
	std::normal_distribution<double> dis(0, 1);
	for (int i = 0; i < backunits_len; i++)
		for (int j = 0; j < units_len; j++)
			weight(i, j) = dis(gen);
	for (int j = 0; j < units_len; j++)
		bias(0, j) = dis(gen);

	cout << "Initialize a layer " << backunits_len << " to " << units_len << "!" << endl;
}

Eigen::MatrixXd DenseLayer::ForwardPropagation(const Eigen::MatrixXd &_x_data)
{
	x_data = _x_data;
	if (is_input_layer == true)
	{
		return x_data;
	}
	else
	{
		wx_plus_b = x_data * weight - bias;
		if (act_func == Activation::Sigmoid)
			output = wx_plus_b.unaryExpr([](double x) { return sigmoid(x); });
		else if (act_func == Activation::ReLu)
			output = wx_plus_b.unaryExpr([](double x) { return relu(x); });
		else if (act_func == Activation::None)
			output = wx_plus_b;
		return output;
	}
}


Eigen::MatrixXd DenseLayer::cal_gradient()
{
	// Calculate a diagnal matrix to represent 1{wx_plus_b[i]>=0}, return a  units_len * units_len matrix.
	Eigen::Matrix<double, 1, Eigen::Dynamic> D;
	if (act_func == Activation::Sigmoid)
		D = wx_plus_b.unaryExpr([](double x) { return Dsigmoid(x); });
	else if (act_func == Activation::ReLu)
		D = wx_plus_b.unaryExpr([](double x) { return Drelu(x); });
	else if (act_func == Activation::None)
		D = wx_plus_b.unaryExpr([](double x) { return 1; });
	return D.asDiagonal();

}


Eigen::MatrixXd DenseLayer::BackwardPropagation(const Eigen::MatrixXd &gradient)
{
	//partial loss/ partial wij= 1{wx_plus_b[i]>=0} * xdatai * gradientj
	Eigen::MatrixXd gradient_activation = cal_gradient();

	gradient_weight = x_data.transpose()*gradient*gradient_activation; //(backunits,1)*(1,units)*(units,units)
	gradient_b = -gradient * gradient_activation; //(1,units)*(units,units)

												  //if(o==Optimizer::GD)
												  //{

												  //}

	weight = weight - learning_rate * gradient_weight;
	bias = bias - learning_rate * gradient_b;

	gradient_to_prop = gradient * (weight*gradient_activation).transpose(); //(1,units)*[(backunits,units)*(units,units)].T

	return gradient_to_prop;

}


class BPNN
{
public:
	BPNN();
	~BPNN();
	void AddLayer(DenseLayer *layer);
	void AddLayer(int _backunits_len, int _units_len, string name, double _learning_rate, bool _is_input_layer, DenseLayer::Activation t);
	void BuildLayer();
	void Summary();
	double Train(const Eigen::MatrixXd& xdata, const Eigen::MatrixXd& ydata, int _train_round, double _accuracy, int validnum);
	Eigen::MatrixXd Predict(const Eigen::MatrixXd& xdata, int output_len);
	void Compare(const Eigen::MatrixXd& xdata, const Eigen::MatrixXd& ydata, int num);
	result Cal_loss(const Eigen::MatrixXd& xdata, const Eigen::MatrixXd& ydata);
	// Add scalor


private:
	vector<DenseLayer*> layers;
	vector<double> train_mse;
	vector<double> valid_mse;
	Eigen::MatrixXd loss_gradient;
	int train_round;
	double accuracy;
};

BPNN::BPNN()
{

}

BPNN::~BPNN()
{
	for (auto layer : layers)
		delete layer;
}

void BPNN::AddLayer(DenseLayer *layer)
{
	layers.push_back(layer);
}

void BPNN::AddLayer(int _backunits_len, int _units_len, string name, double _learning_rate = 0.03, bool _is_input_layer = false, DenseLayer::Activation t = DenseLayer::Sigmoid)
{
	DenseLayer *layer = new DenseLayer(_backunits_len, _units_len, name, _learning_rate, _is_input_layer, t);
	layers.push_back(layer);
}

void BPNN::BuildLayer()
{
	for (int i = 0; i<layers.size(); i++)
	{
		if (i == 0)
			layers[i]->setinputlayer();
		layers[i]->Initializer();
	}
}

void BPNN::Summary()
{
	for (int i = 0; i<layers.size(); i++)
	{
		cout << "-------------" << i << "th layer-------------" << endl;
		cout << "weight shape = " << layers[i]->getbackunits() << "*" << layers[i]->getunits() << endl;
	}
}

double BPNN::Train(const Eigen::MatrixXd& xdata, const Eigen::MatrixXd& ydata, int _train_round, double _accuracy, int validnum = 3)
{
	train_round = _train_round;
	accuracy = _accuracy;
	map<string, Eigen::MatrixXd> weight_log;
	map<string, Eigen::MatrixXd> bias_log;
	map<string, vector<double>> mse_log;
	map<string, Eigen::MatrixXd> validresult_log;
	double to_return = 0;

	int n = xdata.rows();
	double loss = 0;
	double all_loss = 0;
	Eigen::MatrixXd _xdata;
	Eigen::MatrixXd _ydata;

	if (n != ydata.rows())
	{
		cout << "Bad input data!" << endl;
		return 0;
	}

	Eigen::MatrixXd xdatatrain = xdata.block(0, 0, xdata.rows() - validnum, xdata.cols());
	Eigen::MatrixXd ydatatrain = ydata.block(0, 0, ydata.rows() - validnum, ydata.cols());
	Eigen::MatrixXd xdatavalid = xdata.block(xdata.rows() - validnum, 0, validnum, xdata.cols());
	Eigen::MatrixXd ydatavalid = ydata.block(ydata.rows() - validnum, 0, validnum, ydata.cols());

	cout << "Initial mse on training set is " << Cal_loss(xdatatrain, ydatatrain).mse << endl;
	Compare(xdatatrain, ydatatrain, 3);

	cout << "Initial mse on validation set is " << Cal_loss(xdatavalid, ydatavalid).mse << endl;
	Compare(xdatavalid, ydatavalid, 3);

	ofstream train_mse_file, valid_mse_file;
	train_mse_file.open("train_mse.txt", ios::app);
	valid_mse_file.open("valid_mse.txt", ios::app);

	for (int i = 0; i < train_round; i++)
	{
		all_loss = 0;
		for (int j = 0; j < xdatatrain.rows(); j++)
		{
			_xdata = xdatatrain.row(j);
			_ydata = ydatatrain.row(j);

			for (auto layer : layers)
			{
				_xdata = layer->ForwardPropagation(_xdata);
			}

			loss_gradient = 2.0 * (_xdata - _ydata);
			loss = loss_gradient.unaryExpr([](double x) { return x * x / 4.0; }).sum();

			all_loss += loss;
			

			for (int k = 0; k < layers.size() - 1; k++)
			{
				loss_gradient = layers[layers.size() - 1 - k]->BackwardPropagation(loss_gradient);
			}
		}

		double mse = all_loss / xdatatrain.rows();
		train_mse.push_back(mse);
		/*if (abs(train_mse[train_mse.size() - 2] - train_mse[train_mse.size() - 1]) < accuracy)
		{
		cout << "Satisfy accuracy!" << endl;
		return mse;
		}*/
		result valid_res = Cal_loss(xdatavalid, ydatavalid);
		Eigen::MatrixXd valid_pred = valid_res.pred;
		double mse_valid = valid_res.mse;
		valid_mse.push_back(mse_valid);

		// record mse
		train_mse_file << mse << " ";
		valid_mse_file << mse_valid << " ";

		if (i % 100 == 0) {
			ofstream file;
			// record validation data result
			file.open("valid_result.txt", ios::app);
			assert(file.is_open());
			file << "step" << i << endl << valid_pred << endl;
			file.close();

			// record weight and bias
			for (auto layer : layers) {
				weight_log[layer->getname()] = layer->getweights();
				bias_log[layer->getname()] = layer->getbias();
			}
			file.open("model.txt", ofstream::trunc);
			assert(file.is_open());
			file << "step" << i << endl;
			file << "weight: " << endl;
			for (const auto &i : weight_log) {
				file << i.first << ": " << i.second << endl;
			}
			file << "bias: " << endl;
			for (const auto &i : bias_log) {
				file << i.first << ": " << i.second << endl;
			}
			file.close();
		}

		cout << "------------- Finished training round " << i << " -------------" << endl;
		cout << "mse on training set is " << mse << endl;
		cout << "mse on validation set is " << mse_valid << endl;

		cout << "Training set example:" << endl;
		Compare(xdatatrain, ydatatrain, 3);

		cout << "Validation set example:" << endl;
		Compare(xdatavalid, ydatavalid, 3);

		if (mse < accuracy)
		{
			cout << "Satisfy accuracy!" << endl;
			to_return = mse;
			break;
		}

	}
	mse_log["train"] = train_mse;
	mse_log["valid"] = valid_mse;
	
	train_mse_file.close();
	valid_mse_file.close();

	// IO
	ofstream file("valid_result.txt");
	assert(file.is_open());
	for (const auto &i : validresult_log) {
		file << i.first << endl << i.second << endl;
	}
	file.close();

	ofstream file2("model.txt");
	assert(file2.is_open());
	file << "weight: " << endl;
	for (const auto &i : weight_log) {
		file << i.first << ": " << i.second << endl;
	}
	file << "bias: " << endl;
	for (const auto &i : bias_log) {
		file << i.first << ": " << i.second << endl;
	}
	file.close();

	return to_return;
}

result BPNN::Cal_loss(const Eigen::MatrixXd& xdata, const Eigen::MatrixXd& ydata)
{
	double all_loss = 0;
	Eigen::MatrixXd _xdata(1, xdata.cols());
	Eigen::MatrixXd pred(xdata.rows(), ydata.cols());

	for (int j = 0; j < xdata.rows(); j++)
	{
		_xdata = Predict(xdata.row(j), ydata.cols());
		all_loss += (_xdata - ydata.row(j)).unaryExpr([](double x) { return x * x; }).sum();
		pred.row(j) = _xdata;
	}
	return result{ all_loss / xdata.rows(), pred };
}

void BPNN::Compare(const Eigen::MatrixXd& xdata, const Eigen::MatrixXd& ydata, int num)
{
	for (int k = 0; k < num; k++)
	{
		cout << "Predict value of sample " << k << " is " << Predict(xdata.row(k), ydata.cols()) << ". ";
		cout << "Real value is " << ydata.row(k) << ". ";
		cout << "Difference is " << Predict(xdata.row(k), ydata.cols()) - ydata.row(k) << ". " << endl;
	}
}

Eigen::MatrixXd BPNN::Predict(const Eigen::MatrixXd& xdata, int output_len)
{
	Eigen::MatrixXd ydata(xdata.rows(), output_len);
	Eigen::MatrixXd _xdata;
	for (int j = 0; j < xdata.rows(); j++)
	{
		_xdata = xdata.row(j);
		for (auto layer : layers)
		{
			_xdata = layer->ForwardPropagation(_xdata);
		}
		for (int i = 0; i < output_len; i++)
			ydata(j, i) = _xdata(0, i);
	}
	return ydata;
}

//void test()
//{
//	//test the training process
//
//	double learning_rate = 0.03;
//	DenseLayer::Activation act_fun = DenseLayer::Sigmoid;
//	BPNN modelnew;
//
//	modelnew.AddLayer(10, 10, learning_rate, true, act_fun);
//	modelnew.AddLayer(10, 20, learning_rate, false, act_fun);
//	modelnew.AddLayer(20, 30, learning_rate, false, act_fun);
//	modelnew.AddLayer(30, 2, learning_rate, false, DenseLayer::None);
//	modelnew.BuildLayer();
//	modelnew.Summary();
//
//	Eigen::MatrixXd x(10,10);
//	x << -0.42341286, 0.21779802, -0.54369312, 2.04964989, 1.00671986,
//		0.72770789, 0.2580108, 0.74788435, 1.45180192, 0.86803638,
//		-1.43974545, -1.20253251, -1.24224465, 0.24809309, -0.93821806,
//		1.29316884, -0.50198725, -0.63714213, 0.12479802, 0.91007394,
//		-0.78658784, -1.12794307, -0.77812005, 1.29574899, 0.16750844,
//		-0.70761621, 1.51739084, -1.19870489, -1.53029875, -0.9038248,
//		-0.9756778, 0.66175796, 0.26833978, 1.75458108, 0.15402258,
//		-0.42806397, -0.63166847, 0.19717951, -1.97259133, 0.23806793,
//		0.83755467, -0.37247964, -0.06758306, 0.22669441, -0.1273009,
//		1.47156685, 0.30417944, 1.66046617, 1.0805952, 1.02822416,
//		-1.30650562, 0.66428356, -1.51496519, 0.30665193, -0.95840903,
//		0.69387956, 0.54239419, -0.13788214, 1.14797255, -1.18778428,
//		0.92176127, -0.37185503, -0.51249125, -1.52096541, 0.392217,
//		-1.26853408, -0.23724684, 0.72507058, 0.0810218, 1.20581851,
//		0.55981882, -1.77590695, -1.12788518, -0.02926117, 0.31905083,
//		1.11389359, -0.56559586, 0.10578212, -1.30172802, 1.84858769,
//		-0.2738502, -1.44412151, 0.7872747, -0.10611829, 1.06023464,
//		-0.12080409, -1.38991104, -0.51387999, 0.9472472, 0.28645597,
//		0.10045478, 0.2806141, 0.12326028, 0.5001843, 0.22650803,
//		-0.66142985, -0.50764307, 1.35874742, -0.54401188, 1.11425037;
//	Eigen::MatrixXd y(10,2);
//	y << 0.8, 0.4, 0.4, 0.3, 0.34, 0.45, 0.67, 0.32,
//		0.88, 0.67, 0.78, 0.77, 0.55, 0.66, 0.55, 0.43, 0.54, 0.1,
//		0.1, 0.5;
//	modelnew.Train(x, y, 1000, 0.01);
//}

void readdata(string filename, Eigen::MatrixXd& xdata, Eigen::MatrixXd& ydata)
{
	ifstream infile;
	infile.open(filename);
	assert(infile.is_open());

	string date;
	double data;
	for (int i = 0; i < stock_rows; i++)
	{
		//infile >> date;
		//cout << date << endl;
		for (int j = 0; j < stock_cols; j++)
		{
			infile >> data;
			xdata(i, j) = data;
		}
		infile >> data;
		//cout << data << endl;
		ydata(i, 0) = data;
	}

	infile.close();

	cout << "Successfully read the data from " + filename + "!" << endl;
}

//main problem
//1. Gradient Exploding
// pretraining, cutting
//2. don't work, structrue is bad
// no solution
// yanerdaoling

// Scaling
// Yiyao Chen
// add scalor
// change Predict

// momentum
// Ludi Wang

// IOinfrastructure
// save load
// override: weights, valid result (validation result)
// append: learning curve
// model("name")
// "modelname_activation_layeri.txt"
// load

void real_train()
{
	Eigen::MatrixXd xdata(stock_rows, stock_cols);
	Eigen::MatrixXd ydata(stock_rows, 1);
	readdata("SPXdatanormshuffle.txt", xdata, ydata);

	double learning_rate = 0.003;
	DenseLayer::Activation act_fun = DenseLayer::Sigmoid;
	BPNN modelnew;


	modelnew.AddLayer(stock_cols, stock_cols, "input", learning_rate, true, act_fun);
	modelnew.AddLayer(stock_cols, 50, "layer1", learning_rate, false, act_fun);
	modelnew.AddLayer(50, 5, "layer2", learning_rate, false, act_fun);
	modelnew.AddLayer(5, 1, "output", learning_rate, false, DenseLayer::None);	// output layer
	modelnew.BuildLayer();
	modelnew.Summary();

	

	modelnew.Train(xdata, ydata, 100000, 1e-10, 30); // xdata, ydata, train_round

	
}

int main()
{
	//train on real data
	real_train();

	//test the training process
	//test();

	system("pause");
	return 0;
}
