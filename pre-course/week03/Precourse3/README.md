# Moreh-SNU GPU Enginner Training Precourse

## Precourse 3

### Installation

- GNU compiler
- Python 3

### Usage

- Problem 1

```bash
# For list problem
cd Problem1/list
make run
# For vector problem
cd Problem2/vector
make run
```

- Problem 2

```bash
cd Problem2
gcc main.cpp -o main
./main
```

- Problem 3

```bash
cd Problem3
# Create a python virtual enviroment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Python_Standard_Library_PATH=$(python -c "import sys ; print('\n'.join(sys.path))")
c++ -O3 -Wall -shared -std=c++11 -fPIC \
    -I Python_Standard_Library_PATH $(python3 -m pybind11 --includes) \
    matmul.cpp -o matmul_c$(python3-config --extension-suffix)

cp matmul_c.cpython-310-x86_64-linux-gnu.so .venv/lib/python3.10/site-packages

python main.py

deactivate

python main.py
```

- Problem 4

```bash
cd Problem4
make
./kmeans
```

### Authors

Tien M. Pham - email: tien.pham@moreh.com.vn

### License

[MIT](https://choosealicense.com/licenses/mit/)
