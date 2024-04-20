# Détection massive de Fake news à partir de la base de données GDELT

Create a heterogenous graph using the GDELT database containing the following nodes: article, source, and event.
We then train a graph neural network on it to predict fake news on article nodes or source nodes, depending on the graph we create.

### How to use

Install necessary packages:

```
pip install -r requirement.txt
```
To train the model, you need to create the labels and import the GDELT data using the import.py file.

Train the model: 
```
python main.py
```

You can change some parameters (number of csv imported, label type...)


