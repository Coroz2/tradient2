import neptune

def train_model():
    run = neptune.init_run(
        project="dylanad2/CS222",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxY2EzZGZhZS1mMjNlLTQzMGYtOWI3NC1jMTE5OTQzYmQzZTAifQ==",
    )

    params = {"learning_rate": 0.001, "optimizer": "Adam"}
    run["parameters"] = params

    for epoch in range(10):
        run["train/loss"].append(0.9 ** epoch)

    run["eval/f1_score"] = 0.66

    run.stop()

if __name__ == "__main__":
    train_model()