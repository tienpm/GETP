#include "list.h"

// Explicit instantiation
template class getp::list<int>;
template class getp::list<float>;
template class getp::list<double>;

namespace getp {

// Node implementation
template <class T>
list<T>::Node::Node(const T &value) {
    // TODO : FILL IN HERE
    data = value;
    next = nullptr;
    prev = nullptr;
}

// iterator implementation
template <class T> list<T>::iterator::iterator(Node *node) {
    // TODO : FILL IN HERE
    current = node;
}

template <class T> T &list<T>::iterator::operator*() const {
    // TODO : FILL IN HERE
    return current->data;
}

template <class T> typename list<T>::iterator &list<T>::iterator::operator++() {
    // TODO : FILL IN HERE
    if (current != nullptr and current->next != nullptr)
        current = current->next;
    
    return *this;
}

template <class T>
typename list<T>::iterator list<T>::iterator::operator++(int) {
    // TODO : FILL IN HERE
    iterator tmp = *this;
    if (current != nullptr and current->next != nullptr)
        current = current->next;

    return tmp;
}

template <class T>
bool list<T>::iterator::operator!=(const iterator &other) const {
    // TODO : FILL IN HERE
    return current != other.current;
}

// Rest of the list class implementation
template <class T> list<T>::list(){
    // TODO : FILL IN HERE
    head = nullptr;
    tail = nullptr;
    size_ = 0;
}

template <class T>
list<T>::list(const list &other) {
    // TODO : FILL IN HERE
    head = nullptr;
    tail = nullptr;
    size_ = 0;
    Node *it = other.head;
    while (it != other.tail) {
        push_back(it->data);
        it = it->next;
    }
}

template <class T>
list<T>::list(std::initializer_list<T> initList) {
    // TODO : FILL IN HERE
    head = nullptr;
    tail = nullptr;
    size_ = 0;
    for (const T& value : initList) {
        push_back(value);
    }
}

template <class T> list<T>::~list() { 
    // TODO : FILL IN HERE
    clear();
}

template <class T> list<T> &list<T>::operator=(const list &other) {
    // TODO : FILL IN HERE
    if (this != &other) {
        clear();
        Node *it = other.head;
        for (;it != other.tail; it = it->next) {
            push_back(it->data);
        }
    }
     
    return *this;
}

template <class T> void list<T>::push_back(const T &value) {
    // TODO : FILL IN HERE
    Node *newNode = new Node(value);
    if (size_ == 0) {
        head = newNode;
        tail = new Node(0);
        head->next = tail;
        tail->prev= head;
    } else {
        newNode->prev = tail->prev;
        tail->prev->next = newNode;
        newNode->next = tail;
        tail->prev = newNode;
    }
    
    size_++;
}

template <class T> void list<T>::push_front(const T &value) {
    // TODO : FILL IN HERE
    Node *newNode = new Node(value);
    if (head == nullptr) {
        head = newNode;
        tail = new Node(0);
        head->next = tail;
        tail->prev = head;
    } else {
        head->prev = newNode;
        newNode->next = head;
        head = newNode;
    }
    
    size_++;
}

template <class T> void list<T>::pop_back() {
    // TODO : FILL IN HERE
    assert(size_ > 0);
    Node *temp = tail->prev;
    if (size_ == 1) {
        head = tail;
    } else {
        temp->prev->next = tail;
        tail->prev = temp->prev;
    }

    delete temp;
    size_--;
}

template <class T> void list<T>::pop_front() {
    // TODO : FILL IN HERE
    assert(size_ > 0);
    Node *temp = head;
    head = head->next; // = tail if size_ = 1
    delete temp;
    size_--;
}

template <class T> void list<T>::emplace_back(T &&value) {
    Node* newNode = new Node(std::forward<T>(value)); 
    if (size_ == 0) {
        head = newNode;
        tail = new Node(0);
        head->next = tail;
        tail->prev = head;
    } else {
        newNode->prev = tail->prev;
        tail->prev->next = newNode;
        newNode->next = tail;
        tail->prev = newNode;
    }
    size_++;
}

template <class T> void list<T>::emplace_front(T &&value) {
    // TODO : FILL IN HERE
    Node* newNode = new Node(std::forward<T>(value));
    if (head == nullptr) {
        head = newNode;
        tail = new Node(0);
        head->next = tail;
        tail->prev = head;
    } else {
        head->prev = newNode;
        newNode->next = head;
        head = newNode;
    }
    size_++;
}

template <class T> std::size_t list<T>::size() const {
    // TODO : FILL IN HERE
    return size_;
}

template <class T> void list<T>::print() const {
    // TODO : FILL IN HERE
    Node *current = head;
    while (current != tail) {
        std::cout << current->data << " ";
        current = current->next;
    }
    std::cout << "\n";
}

template <class T> void list<T>::clear() {
    // TODO : FILL IN HERE
    Node *iter = head;
    while (iter != nullptr) {
        Node *tmp = iter;
        iter = iter->next;
        delete tmp;
    }
    head = tail = nullptr;
    size_ = 0;
}

template <class T> typename list<T>::iterator list<T>::begin() {
    // TODO : FILL IN HERE
    return iterator(head);
}

template <class T> typename list<T>::iterator list<T>::end() {
    // TODO : FILL IN HERE
    return iterator(tail);
}

template <class T> typename list<T>::iterator list<T>::cbegin() const {
    // TODO : FILL IN HERE
    return iterator(head);
}

template <class T> typename list<T>::iterator list<T>::cend() const {
    // TODO : FILL IN HERE
    return iterator(tail);
}

template <class T> typename list<T>::iterator list<T>::rbegin() {
    // TODO : FILL IN HERE
    return iterator(tail);
}

template <class T> typename list<T>::iterator list<T>::rend() {
    // TODO : FILL IN HERE
    return iterator(head);
}

template <class T> typename list<T>::iterator list<T>::crbegin() const {
    // TODO : FILL IN HERE
    return iterator(tail);
}

template <class T> typename list<T>::iterator list<T>::crend() const {
    // TODO : FILL IN HERE
    return iterator(head);
}

} // namespace getp
