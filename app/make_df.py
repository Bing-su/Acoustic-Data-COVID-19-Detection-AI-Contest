import pandas as pd


def id_to_audio_train(id: int):
    return "data/train/" + str(id).zfill(5) + ".wav"


def id_to_audio_test(id: int):
    return "data/test/" + str(id).zfill(5) + ".wav"


train_df = pd.read_csv("data/train_data.csv")
test_df = pd.read_csv("data/test_data.csv")

train_df["audio"] = train_df["id"].apply(id_to_audio_train)
test_df["audio"] = test_df["id"].apply(id_to_audio_test)

train_age_mean = train_df["age"].mean()
train_age_std = train_df["age"].std()

train_df["age_scaled"] = (train_df["age"] - train_age_mean) / train_age_std
test_df["age_scaled"] = (test_df["age"] - train_age_mean) / train_age_std

gender_map = {"female": 0, "male": 1, "other": 0}
train_df["gender"].replace(gender_map, inplace=True)
test_df["gender"].replace(gender_map, inplace=True)

order = [
    "audio",
    "age_scaled",
    "gender",
    "respiratory_condition",
    "fever_or_muscle_pain",
    "covid19",
]
train_df = train_df.loc[:, order]
order.pop()
test_df = test_df.loc[:, order]

train_df.to_csv("data/train_df.csv", index=False)
test_df.to_csv("data/test_df.csv", index=False)
