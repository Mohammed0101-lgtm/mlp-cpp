#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <cassert>
#include <sstream>

// hyperparams
const int embd_dims;
const int block_size;
const int layer_size = 100;
const int batch_size;
const int eval_iters;
const int max_iterations;
const int eval_interval;

class DataLoader {
private:
    std::unordered_map<int, char> global_idx;
    std::unordered_map<char, int> char_to_index;
    int vocab_size = 0;
public:
    DataLoader() {}

    int get_vocab_size() { return global_idx.size(); }

    std::vector<int> encode(const std::string& str) {
        std::vector<int> result;
        char_to_index.clear();
        int index = 0;

        for (char c : str) {
            if (char_to_index.find(c) == char_to_index.end()) 
                char_to_index[c] = index++;

            result.push_back(char_to_index[c]);
        }

        global_idx.clear(); 
        for (const auto& pair : char_to_index) 
            global_idx[pair.second] = pair.first;

        return result;
    }

    std::string decode(data::Tensor<data::Value>& tensor) const {
        std::string result;
        result.reserve(tensor.size(0)); 

        for (size_t i = 0; i < tensor.size(0); ++i) {
            int index = static_cast<int>(tensor.get(std::vector<int>(i)).data);
            if (global_idx.find(index) != global_idx.end()) 
                result.push_back(global_idx.at(index));
        }
        return result;
    }

    std::string read_file(const std::string& filename) {
        std::ifstream file(filename, std::ios::in);
        if (!file.is_open()) 
            throw std::runtime_error("Failed to open file for read");

        std::stringstream buf;
        buf << file.rdbuf();
        file.close();
        return buf.str();
    }       

    std::tuple<data::Tensor<int>,data::Tensor<int>> get_batch(const data::Tensor<int>& data) {

    }

    std::tuple<std::vector<int>, std::vector<int>> split_data(const std::vector<int>& data) {
        size_t size = static_cast<size_t>(data.size() * 0.9);
        std::vector<int> train_data(data.begin(), data.begin() + size);
        std::vector<int> val_data(data.begin() + size, data.end());
        return std::make_tuple(train_data, val_data);
    }
};


namespace data {

class Value {
public:
    float data;
    float grad;
    int _op;
    std::vector<std::shared_ptr<Value>> _prev;

    Value() : data(0.0f), grad(0.0f), _op(' '), _prev() {}
    Value(float val) : data(val), grad(0.0f), _op(' '), _prev() {}
    Value(Value& val) : data(val.data), grad(val.grad), _op(val._op), _prev(val._prev) {}
    Value(const Value& val) : data(val.data), grad(val.grad), _op(val._op), _prev(val._prev) {}
    Value(float val, char operation, std::vector<std::shared_ptr<Value>>& children) 
      : data(val), grad(0.0f), _op(operation), _prev(children) {}

    Value operator+(const Value& other) {
        std::vector<std::shared_ptr<Value>> p;
        p.push_back(std::make_shared<Value>(*this));
        p.push_back(std::make_shared<Value>(other));
        return Value(this->data + other.data, '+', p);
    }

    Value operator-(const Value& other) {
        std::vector<std::shared_ptr<Value>> p;
        p.push_back(std::make_shared<Value>(*this));
        p.push_back(std::make_shared<Value>(other));
        return Value(this->data - other.data, '-', p);
    }

    Value operator*(const Value& other) {
        std::vector<std::shared_ptr<Value>> p;
        p.push_back(std::make_shared<Value>(*this));
        p.push_back(std::make_shared<Value>(other));
        return Value(this->data * other.data, '*', p);
    }

    void _backward() {
        switch (_op) {
            case '+':
                _prev[0]->grad += grad;
                _prev[1]->grad += grad;
                break;
            case '*':
                _prev[0]->grad += grad * _prev[1]->data;
                _prev[1]->grad += grad * _prev[0]->data;
                break;
            case '^':
                float exponent = _prev[1]->data;
                _prev[0]->grad += exponent * std::pow(_prev[0]->data, exponent - 1) * grad;
                break;
            case 'r':
                _prev[0]->grad += (_prev[0]->data > 0) * grad;
                break;
            default :
                throw std::runtime_error("operation not supported"); 
        }
    }
};

template<typename T>
class Tensor {
public:
    std::vector<T> data;
    std::vector<int> strides;
    std::vector<int> shape;
    
    Tensor() {}
    Tensor(Tensor<T> tens) : data(tens.data), strides(tens.strides), shape(tens.shape) {}
    Tensor(std::vector<T> d) : data(d) { computeStrides(); } 
    Tensor(std::vector<int> sh) : shape(sh) { computeStrides(); }
    Tensor(std::vector<T> d, std::vector<int> sh) : data(d), shape(sh) { computeStrides(); }

    T& get(std::vector<int> indices) { return data[getIndex(indices)]; }
    void set(std::vector<int> indices, T val) { data[getIndex(indices)] = val; }

    size_t size(int dim) const {
        if (dim < 0 || static_cast<size_t>(dim) >= shape.size()) 
            throw std::out_of_range("Dimension out of range");

        if (dim == -1) return data.size(); 
        return shape[dim];  
    }

    std::vector<T> get_row(int index) {
        assert(shape.size() == 2);
        assert(row_index >= 0 && row_index < shape[0]);
        int start_idx = row_index * shape[1];
        std::vector<Value> row;
        row.reserve(shape[1]);
        
        for (int i = 0; i < shape[1]; i++) 
            row.push_back(data[start_idx + i]); 

        return row;
    }

    void set_row(int idx, std::vector<T> row) {
        assert(shape.size() == 2);
        assert(idx >= 0 && idx < shape[0]);
        assert(row.size() == shape[1]);
        int start_idx = idx * shape[1];
    
        for (int i = 0; i < shape[1]; i++) 
            data[start_idx + i] = row[i];
    }

private:
    void computeStrides() {
        strides.resize(shape.size());
        int stride = 1;
        for (int i = 0; i < strides.size(); i++) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

    int getIndex(std::vector<int> indices) const {
        if (indices.size() != shape.size()) return -1;

        int index = 0;
        for (size_t i = 0; i < indices.size(); ++i) 
            index += indices[i] * strides[i];
    
        return index;
    }
};


Tensor<Value> matmul(Tensor<Value> m1, Tensor<Value> m2) {
    assert(m1.shape.size() == 2 && m2.shape.size() == 2);
    assert(m1.shape[1] == m2.shape[0]);

    Tensor<Value> result(static_cast<std::vector<int>>(m1.shape[0], m2.shape[1]));
    for (int i = 0; i < m1.shape[0]; i++) 
        for (int j = 0; j < m2.shape[1]; j++) {
            float sum = 0.0f;
            for (int k = 0; k < m1.shape[1]; k++) {
                int a[2] = {i, k};
                int _a[2] = {k, j};
                sum +=  (m1.get(std::vector<int>(a, a + 2)) * m2.get(std::vector<int>(std::vector<int>(_a, _a + 2)))).data;
            }

            int a[2] = {i, j};
            result.set(std::vector<int>(a, a + 2), Value(sum));
        }

    return result;
}

Tensor<Value> relu(Tensor<Value>& x) {
    size_t size = x.size(-1);
    for (size_t i = 0; i < size; i++) 
        x.data[i].data = std::max(x.data[i].data, 0.0f);

    return x;
}

Tensor<Value> cross_entropy(Tensor<Value> predicted, Tensor<Value> targets) {

}

}; // data


namespace nn {

class Embedding {
public:
    data::Tensor<data::Value> table;  

    Embedding() {}
    Embedding(std::vector<int> dims) : table(dims) {}

    void _init_(const std::vector<int>& dims) {
        table = data::Tensor<data::Value>(dims);  
        size_t size = table.size(-1);
        std::vector<data::Value> elements;
        
        for (size_t i = 0; i < size; i++) 
            elements.push_back(data::Value(static_cast<double>(rand()) / RAND_MAX));

        table.data = elements;  
    }

    data::Tensor<data::Value> forward(data::Tensor<int>& input) {
        std::vector<int> indices(input.data);   
        int d[2] = {static_cast<int>(indices.size()), table.shape[1]};
        std::vector<int> output_dims(d, d + 2);
        data::Tensor<data::Value> output(output_dims);  
    
        for (size_t i = 0; i < indices.size(); i++) {
            int idx = indices[i];
            std::vector<data::Value> embedding = table.get_row(idx);  
            output.set_row(i, embedding);  
        }

        return output;  
    }

    void backward(data::Tensor<data::Value>& input, data::Tensor<data::Value>& grad_output) {
        std::vector<int> indices;  
        for (data::Value& val : input.data) {
            indices.push_back(static_cast<int>(val.data));
        }

        std::vector<data::Value> table_data = table.data;
        int cols = table.shape[1];  
        for (size_t i = 0; i < indices.size(); i++) {
            int idx = indices[i];
            std::vector<data::Value> grad = grad_output.get_row(i);  

            for (size_t j = 0; j < grad.size(); j++) {
                int table_index = idx * cols + j;
                table_data[table_index].grad += grad[j].grad;
            }
        }
    
        table.data = table_data;
    }
};

class Linear {
public:
    data::Tensor<data::Value> weights;
    data::Tensor<data::Value> biases;
    bool bias;

    Linear() : bias(false) {}
    Linear(std::vector<int> dims) : weights(dims), biases(std::vector<int>(dims[0] * dims[1])), bias(true) {}
    
    bool is_biased() { return bias; }
    data::Tensor<data::Value> get_weights() { return weights; }

    data::Tensor<data::Value> forward(data::Tensor<data::Value> input) {
        input = data::matmul(input, weights);

        if (bias) 
            for (size_t i = 0, size = input.size(-1); i < size; i++) 
                input.data[i].data += biases.data[i].data;

        input = data::relu(input);
        return input;
    }
};

class MLP {
private:
    std::vector<data::Value> parameters;
    Embedding lookup_table;
    Linear hidden_layer;
    Linear output_layer;
    bool traning;

    MLP(int vocab_size) 
    {
        std::vector<int> input_dims;
        input_dims.push_back(vocab_size);
        input_dims.push_back(embd_dims);
        lookup_table = Embedding(input_dims);   
        
        std::vector<int> hidden_dims;
        hidden_dims.push_back(block_size * embd_dims);
        hidden_dims.push_back(layer_size);
        hidden_layer = Linear(hidden_dims);

        std::vector<int> output_dims;
        output_dims.push_back(layer_size);
        output_dims.push_back(vocab_size);
        output_layer = Linear(output_dims);
    }

    data::Tensor<data::Value> feed_forward(data::Tensor<int> input) {
        data::Tensor<data::Value> logits = lookup_table.forward(input);
        
        if (input.size(1) != block_size * embd_dims) {
            std::cerr << "Error: Mismatch in input dimension after embedding. "
                      << "Expected " << block_size * embd_dims << ", got " << input.size(1) << std::endl;
            std::exit(EXIT_FAILURE); 
        }

        logits = hidden_layer.forward(logits);
        logits = data::relu(logits);
        logits = output_layer.forward(logits);
        return logits;
    }

    std::unordered_map<std::string, float> estimate_loss(
        DataLoader& data_loader, 
        data::Tensor<int> train_data, 
        data::Tensor<int> val_data
    ) {
        std::unordered_map<std::string, float> loss_map;  

        for (const std::string& split : {"train", "val"}) {
            float total_loss = 0.0;  
            data::Tensor<int> data = (split == "train") ? train_data : val_data;
            
            for (int k = 0; k < eval_iters; k++) {
                auto [inputs, targets] = data_loader.get_batch(data);  
                data::Tensor predicted = feed_forward(inputs);  
                data::Tensor loss = data::cross_entropy(predicted, targets);  
                total_loss += loss.data[k].data;  
            }

            float avg_loss = total_loss / static_cast<float>(eval_iters);
            loss_map[split] = avg_loss;  
        }

        return loss_map;  
    }

    void train_model(DataLoader& loader, const std::vector<int>& data) {
        size_t data_size = data.size();
        auto [train_data, val_data] = loader.split_data(data);
        data::Tensor<int> train_tensor = data::Tensor<int>(train_data);
        data::Tensor<int> val_tensor = data::Tensor<int>(val_data);
        for (int iter = 0; iter < max_iterations; iter++) 
            if (iter % eval_interval == 0) {
                auto losses = estimate_loss(loader, train_tensor, val_tensor);
                std::cout << "step " << iter << ": Train loss = " << losses["train"] << " , Val loss = " << losses["val"] << std::endl;
            }

        auto [inputs, targets] = loader.get_batch(train_tensor);

    }

};

};

int main() {
    return 0;
}
