#pragma once

#include <iostream>
#include <iterator>
#include <stdexcept>
#include <queue>

namespace getp {

template <class T>
class priority_queue {
private:
  // IMPLEMENT HERE
  std::vector<T> data;

  void heapify_up(int index) {
    while (index > 0) {
      int parent = (index - 1) / 2;
      if (data[index] > data[parent]) { // Maintain max-heap property
        std::swap(data[index], data[parent]);
        index = parent;
      } else {
        break;
      }
    }
  }

  void heapify_down(int index) {
    int left = 2 * index + 1;
    int right = 2 * index + 2;
    int largest = index;

    if (left < (int)data.size() && data[left] > data[largest]) {
      largest = left;
    }
    if (right < (int)data.size() && data[right] > data[largest]) {
      largest = right;
    }

    if (largest != index) {
      std::swap(data[index], data[largest]);
      heapify_down(largest);
    }
  }

public:
  priority_queue();
  ~priority_queue();
  void push( const T& value );
  const T& top() const;
  void pop();
};

template <class T>
priority_queue<T>::priority_queue() {
  // IMPLEMENT HERE
}

template <class T>
priority_queue<T>::~priority_queue() { 
  // IMPLEMENT HERE
  data.clear();
}

template <class T>
void priority_queue<T>::push(const T &value) {
  // IMPLEMENT HERE
  data.push_back(value);
  // Heapify up to restore heap property
  heapify_up(data.size()-1);
}

template <class T>
const T& priority_queue<T>::top() const {
  // IMPLEMENT HERE
  if (data.empty()) {
    throw std::runtime_error("Priority queue is empty");
  }
  return data[0];
}

template <class T>
void priority_queue<T>::pop() {
  // IMPLEMENT HERE
  if (data.empty()) {
    throw std::runtime_error("Priority queue is empty");
  }
    std::swap(data[0], data.back()); // Swap root with last element and remove it
    data.pop_back(); 
    heapify_down(0); // Restore heap property
}

} // namespace getp
