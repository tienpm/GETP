class MinHeap:
    def __init__(self):
        self.heap = []

    def _parent(self, idx):
        return (idx - 1) // 2

    def _left(self, idx):
        return 2 * idx + 1

    def _right(self, idx):
        return 2 * idx + 2

    def _has_left(self, idx):
        return self._left(idx) < len(self.heap)

    def _has_right(self, idx):
        return self._right(idx) < len(self.heap)

    def _swap(self, i, j):
        '''
            Swap the elements at indices i and j of heap array
        '''
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def _upheap(self, idx):
        parent = self._parent(idx)
        if idx > 0 and self.heap[idx] < self.heap[parent]:
            self._swap(idx, parent)
            self._upheap(parent)

    def _downheap(self, idx):
        if self._has_left(idx):
            left = self._left(idx)
            small_child = self._left(idx)
            if self._has_right(idx):
                right = self._right(idx)
                if self.heap[right] < self.heap[left]:
                    small_child = right

            if self.heap[small_child] < self.heap[idx]:
                self._swap(idx, small_child)
                self._downheap(small_child)

    def is_empty(self):
        return len(self.heap) == 0
 
    def __len__(self):
        '''
            Return the number of item in the heap
        '''
        return len(self.heap)

    def push(self, value):
        # TODO : FILL IN HERE
        self.heap.append(value)                 # Add new node at the right-most leaf of the tree
        self._upheap(len(self.heap) -  1)       # Find and re-order the complete binary tree

    def pop(self):
        # TODO : FILL IN HERE
        if self.is_empty():
            raise IndexError("Can not pop because heap is empty")
        self._swap(0, len(self.heap) - 1)       # put minimum item at the right-most leaf (the end of the list)
        val = self.heap.pop()                   # and remove it from the list
        self._downheap(0)                       # then re-order new root (the right-most leaf at begin)
        return val

    def heapify(self):
        # TODO : FILL IN HERE
        start = self._parent(len(self) - 1)     # start with the parent of the right-most leaf
        for i in range(start, -1, -1):          # re-order the complete binary tree
            self._downheap(i)

if __name__ == "__main__":
    min_heap = MinHeap()

    with open('input_heap.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            value = int(line.strip())
            min_heap.push(value)

    print("Min heap : ", min_heap.heap)
