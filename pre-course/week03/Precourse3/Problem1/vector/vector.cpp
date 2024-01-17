#include "vector.h"

namespace getp {

    template <typename T>
    vector<T>::vector() : data_(nullptr), size(0), capacity(0) {}

    template <typename T>
    vector<T>::~vector() {
        delete[] data_;
    }

    template <typename T>
    vector<T>::vector(const vector& other) : data_(new T[other.size]), size(other.size), capacity(other.size) {
        std::copy(other.data_, other.data_ + size, data_);
    }

    template <typename T>
    vector<T>::vector(vector&& other) noexcept : data_(other.data_), size(other.size), capacity(other.capacity) {
        other.data_ = nullptr;
        other.size = 0;
        other.capacity = 0;
    }

    template <typename T>
    vector<T>& vector<T>::operator=(const vector& other) {
        if (this != &other) {
            delete[] data_;
            data_ = new T[other.size];
            size = other.size;
            capacity = other.capacity;
            std::copy(other.data_, other.data_ + size, data_);
        }
        return *this;
    }

    template <typename T>
    vector<T>& vector<T>::operator=(vector&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            size = other.size;
            capacity = other.capacity;
            other.data_ = nullptr;
            other.size = 0;
            other.capacity = 0;
        }
        return *this;
    }

    template <typename T>
    T& vector<T>::at(size_t index) {
        if (index >= size) {
            throw std::out_of_range("Index out of range");
        }
        return data_[index];
    }

    template <typename T>
    T& vector<T>::operator[](size_t index) {
        if (index >= size) {
            throw std::out_of_range("Index out of range");
        }
        return data_[index];
    }

    template <typename T>
    T& vector<T>::front() {
        return data_[0];
    }

    template <typename T>
    T& vector<T>::back() {
        return data_[size - 1];
    }

    template <typename T>
    T* vector<T>::data() {
        return data_;
    }

    template <typename T>
    size_t vector<T>::getSize() const {
        return size;
    }

    template <typename T>
    void vector<T>::push_back(const T& value) {
        if (size == capacity) {
            reserve(capacity == 0 ? 1 : 2 * capacity);
        }
        data_[size++] = value;
    }

    template <typename T>
    template <typename... Args>
    void vector<T>::emplace_back(Args &&... args) {
        if (size == capacity) {
            reserve(capacity == 0 ? 1 : 2 * capacity);
        }
        data_[size++] = T(std::forward<Args>(args)...);
    }

    template <typename T>
    template <typename InputIt>
    void vector<T>::append_range(InputIt first, InputIt last) {
        size_t newSize = size + std::distance(first, last);
        if (newSize > capacity) {
            reserve(newSize);
        }
        std::copy(first, last, data_ + size);
        size = newSize;
    }

    template <typename T>
    void vector<T>::pop_back() {
        if (size > 0) {
            --size;
        }
    }

    template <typename T>
    void vector<T>::reserve(size_t newCapacity) {
        if (newCapacity > capacity) {
            T* newData = new T[newCapacity];
            std::copy(data_, data_ + size, newData);
            delete[] data_;
            data_ = newData;
            capacity = newCapacity;
        }
    }

    template <typename T>
    T* vector<T>::begin() {
        return data_;
    }

    template <typename T>
    const T* vector<T>::cbegin() const {
        return data_;
    }

    template <typename T>
    T* vector<T>::end() {
        return data_ + size;
    }

    template <typename T>
    const T* vector<T>::cend() const {
        return data_ + size;
    }

    template <typename T>
    std::reverse_iterator<T*> vector<T>::rbegin() {
        return std::reverse_iterator<T*>(end());
    }

    template <typename T>
    std::reverse_iterator<const T*> vector<T>::crbegin() const {
        return std::reverse_iterator<const T*>(cend());
    }

    template <typename T>
    std::reverse_iterator<T*> vector<T>::rend() {
        return std::reverse_iterator<T*>(begin());
    }

    template <typename T>
    std::reverse_iterator<const T*> vector<T>::crend() const {
        return std::reverse_iterator<const T*>(cbegin());
    }

    // Explicit instantiation for supported types
    template class vector<int>;
    template void getp::vector<int>::emplace_back<int>(int&&);

}