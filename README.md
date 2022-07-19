# car-price-prediction
you can clone the repository by executing this command:

`git clone git@github.com:abeeralshaer/car-price-prediction.git`

first you must create a conda enviroment using code bellow (I assigned CC to the name of enviroment).
> if you have not conda on your local computer check this [instruction](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

`conda create --name CC`

`conda activate CC`

then install all dependencies using:

`pip install -r requirements.txt`

Then You must download the datasets using this [link](https://www.kaggle.com/datasets/rupeshraundal/marketcheck-automotive-data-us-canada?select=us-dealers-used.csv)
and place them inside dataset's directory.

you can train the model by using these command:

`cd codes`

`python code.py`
