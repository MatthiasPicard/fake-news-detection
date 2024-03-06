import data
import preprocessing
import heterogemodel
import torch
import torch.optim as optim
import torch.nn as nn

num_features = 100
hidden_dim = 64
num_classes = 2
num_relations = 3

def train(model, optimizer, criterion, num_epochs, data, y_true, device):
    model.to(device)
    data = data.to(device)
    y_true = y_true.to(device)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        output = model(data)

        labeled_mask = y_true >= 0
        masked_output = output[labeled_mask]
        masked_y_true = y_true[labeled_mask]
        loss = criterion(masked_output, masked_y_true)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")


def test():
    pass

if __name__ == "__main__":
    df_mentions, df_sources, df_sources_mixte, df_articles_mixte, MBFS, df_test, df_test_event = Data.data_load()
    data = Preprocessing.create_graph(df_sources_mixte,df_test,df_test_event)

    article_features = df_test["MentionIdentifier"].unique()
    source_features = df_sources_mixte["links"].unique()
    event_features = df_test_event["GlobalEventID"].unique()

    num_article_features = len(article_features)
    num_source_features = len(source_features)
    num_event_features = len(event_features)
    num_features = max(num_article_features, num_source_features, num_event_features)

    print("Number of features:", num_features)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_true = data.y.float()

    hidden_dim = 64
    num_classes = 2
    num_relations = 3
    model = HeterogeModel(num_features, hidden_dim, num_classes, num_relations)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    train(model, optimizer, criterion, num_epochs=3, data=data, y_true=y_true, device=device)










