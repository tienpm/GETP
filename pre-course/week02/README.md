# Moreh-SNU GPU Enginner Training Precourse

## Precourse 2

### Installation


Create a Python virtual environment and install the requirement package

```bash
python3 -m venv <virtual_environment_path>
pip install -r requirements.txt
```

Example:
```bash
python3 -m venv .venv
pip install -r requirements.txt
``` 

### Usage

Activate environment
```bash
source <virtual_environment_path>/bin/activate
```
Example:
```bash
source .venv/bin/activate
```

- Problem 1

```bash
cd Problem1
python heap.py
python dijkstra.py
```
- Problem 2
```bash
cd Problem2
python main.py < input.txt
```
- Problem 3
```bash
cd Problem3
python main.py
```
- Problem 4
```bash
cd Problem4
python main.py --number_imgs 4 --filename "sobel_result"
```

### Authors

Tien M. Pham - email: tien.pham@moreh.com.vn

### License

[MIT](https://choosealicense.com/licenses/mit/)