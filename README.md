# Détection massive de Fake news à partir de la base de données GDELT

Create a heterogenous graph using the GDELT database containing the following nodes: article, source, and event.
We then train a graph neural network on it to predict fake news on article nodes or source nodes, depending on the graph we create.

### How to use

Install necessary packages:

```
pip install -r requirement.txt
```
To train an example graph (labeled on articles, contains around 9h of data from the beginning of october, 2023), you need to download it by following this link: https://drive.google.com/file/d/1TcLouPUhwaxTA83Z3RKXzCO_CMZeKqYC/view?usp=sharing.

Then put this file into a directory called "saved_graphs" at the root of the repo.
You can then change some parameters of the model on the main.py file.

Train the model: 
```
python main.py
```
It is more complicated to create a graph from scratch, as you would need to create the labels and import the data from GDELT using the import.py file ( you can't do that easily, you need to modify the file).




