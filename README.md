# Ukrainian Stories For Kids Generation Based On Multilingual Bert

The goal of this final project was to train multilingual Bert from Google on Ukrainian corpus to compare original model with trained version on Masked Language Model and Next Sentence Prediction combined to see how good the original model was and if some improvement could have been made. 

One of the biggest challenges that was faced in this project was to find a suitable dataset. Since Ukrainian corpuses are not widespread it was necessary to create one. The initial guess was that although BERT is claiming to be multilingual, it was not performing well on low-resource languages like Ukrainian. The assumption proved itself to be true as you will be able to see later. Short stories for kids and fairytales are a good candidates for training corpus since they are comprised of not so big of a voabulary and generally have similar narration structure. It's important to mention that vocabulary of a child is not as developed as that of an adult, so the model might do a much better job training on it.

## Google Colab Notebooks
This folder contains two colab notebooks that was used for this project, you can also access them via:

[Inference](https://drive.google.com/open?id=1NplLt-I83_zheEbYFg8tyQmlJAKfQgC4)

[Train](https://drive.google.com/open?id=1saShtNnVbKDXG9P8w2T7cePGqyV9rcwU)
### Note: You might want to consider reading through [Inference](https://drive.google.com/open?id=1NplLt-I83_zheEbYFg8tyQmlJAKfQgC4) notebook first to better understand the project 

## Fairytales

This folder contains train and test stores. Train stories are stored in alphabetical order by folders.
Test stories are useful since you might use them to see how well the model works

## Installation

In order to run this on your local machine please download models from this [drive](https://drive.google.com/drive/folders/1KIme_ZEVpGfx36FQB_O3BL_I9-Eo8mG6?usp=sharing) and place them in the root level in server folder.

To install requirements for server run:

```bash
pip install -r requirements.txt
```

To start backend run:

```bash
python application.py
```

To start client navigate to client folder and run the following in the command line:
```bash
npm start
```
Client runs on port 3000 by default

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
