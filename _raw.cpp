//
// _raw.cpp
//

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>


const int ADD = static_cast<int>('+');
const int SUB = static_cast<int>('-');
const int MUL = static_cast<int>('*');
const int DIV = static_cast<int>('/');
const int POW = static_cast<int>('^');
const int NUL = static_cast<int>(' ');

const int BEGIN_OF_TEXT = -1;
const int END_OF_TEXT   = -2;
int       VOCAB_SIZE;

class Value
{
   public:
    float                               _data_;
    float                               _grad_;
    std::vector<std::shared_ptr<Value>> _parents_;
    int                                 _op_;

    Value() :
        _data_(0.0f),
        _grad_(0.0f),
        _parents_({nullptr}),
        _op_(NUL) {}

    Value(const float _f) :
        _data_(_f),
        _grad_(0.0f),
        _parents_({nullptr}),
        _op_(NUL) {}

    Value(const Value& _v) :
        _data_(_v._data_),
        _grad_(_v._grad_),
        _parents_(_v._parents_),
        _op_(_v._op_) {}

    Value(const float _d, std::vector<std::shared_ptr<Value>>& _p, const int _op) :
        _data_(_d),
        _grad_(0.0f),
        _parents_(_p),
        _op_(_op) {}

    Value operator+(const Value& _other) const {
        std::vector<std::shared_ptr<Value>> parents;
        parents.push_back(std::make_shared<Value>(*this));
        parents.push_back(std::make_shared<Value>(_other));
        return Value(this->_data_ + _other._data_, parents, ADD);
    }

    Value operator-(const Value& _other) const { return *this + Value(-1.0f) * (_other); }

    Value operator*(const Value& _other) const {
        std::vector<std::shared_ptr<Value>> parents;
        parents.push_back(std::make_shared<Value>(*this));
        parents.push_back(std::make_shared<Value>(_other));
        return Value(this->_data_ * _other._data_, parents, MUL);
    }

    Value operator/(const Value& _other) const { return Value(this->_data_ / _other._data_); }

    void _backward() {
        if (this->_parents_.empty())
            return;

        switch (this->_op_)
        {
        case ADD :
            this->_parents_[0]->_grad_ += this->_grad_;
            this->_parents_[1]->_grad_ += this->_grad_;
            return;
        case MUL :
            this->_parents_[0]->_grad_ += this->_parents_[1]->_grad_ * this->_grad_;
            this->_parents_[1]->_grad_ += this->_parents_[0]->_data_ * this->_grad_;
            return;
        case POW :
            this->_parents_[0]->_grad_ +=
              (this->_parents_[1]->_data_ * std::pow(this->_data_, this->_parents_[1]->_data_ - 1)) * this->_grad_;
            return;
        }
    }

    void backward() {
        std::vector<Value> topo;
        std::set<Value>    visited;
        build_topo(std::make_shared<Value>(*this), visited, topo);
        this->_grad_ = 1.0f;
        std::reverse(topo.begin(), topo.end());

        for (Value& v : topo)
            v._backward();
    }

    void build_topo(const std::shared_ptr<Value> _v, std::set<Value>& _visited, std::vector<Value>& _topo) {
        if (std::find(_visited.begin(), _visited.end(), _v) == _visited.end())
        {
            _visited.insert(*_v);
            for (const std::shared_ptr<Value> parent : _v->_parents_)
                build_topo(parent, _visited, _topo);

            _topo.push_back(*_v);
        }
    }
};


class DataLoader
{
   public:
    std::string read_file(const std::string& _filename) const {
        if (_filename.empty())
            throw std::invalid_argument("Filename Cannot be empty!");

        std::ifstream file(_filename, std::ios::in);
        if (!file.is_open())
            throw std::runtime_error("Cannot open file : " + _filename);

        std::stringstream buf;
        file >> buf.rdbuf();
        file.close();
        return buf.str();
    }

    std::vector<int> encode(const std::string& _str) const {
        if (_str.empty())
            throw std::invalid_argument("Cannot encode an empty string");

        std::vector<int>         ret(_str.length() + 2);
        std::unordered_set<char> charset;
        size_t                   i = 0;
        ret[i++]                   = BEGIN_OF_TEXT;

        for (const char& c : _str)
        {
            ret[i++] = static_cast<int>(c);
            charset.insert(c);
        }
        ret[i]     = END_OF_TEXT;
        VOCAB_SIZE = charset.size();
        return ret;
    }

    std::string decode(const std::vector<int>& _tokens) const {
        if (_tokens.empty())
            throw std::invalid_argument("Cannot decode an empty set of tokens");

        std::string ret = "";

        for (const int& t : _tokens)
            ret += static_cast<char>(t);

        return ret;
    }
};


class LookupTable
{
   private:
    std::vector<int> r;

   public:
};


class Linear
{
   private:
    std::vector<std::vector<Value>> _weights_;
    std::vector<Value>              _biases_;

   public:
    Linear() = default;

    Linear(float _size) :
        _weights_(_size),
        _biases_(_size) {}

    std::vector<Value> forward(const std::vector<int>& _input) const {
        std::vector<Value> logits;

        return logits;
    }
};


int main(void) { 
    return 0; 
}

