## Categorizer

#### Installing Dependencies

- pip install -r requirements.txt

### running the code

- download the word vector file from  [here](https://nlp.stanford.edu/projects/glove/) and place it in the same folder. It should be named `glove.6B.200d.txt`


- python categorize.py


*NOTE* to set tensorflow as keras backend edit the .keras/keras.json file with following content

```
{
    "image_dim_ordering": "tf",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```
