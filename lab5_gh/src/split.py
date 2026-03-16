# STRATIFIED SPLIT

seed = 42

train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label_num"],
    random_state=seed
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label_num"],
    random_state=seed
)

print(len(train_df), len(val_df), len(test_df))

# split IDs
train_df["text_id"].to_csv("splits_train_ids.txt", index=False, header=False)
val_df["text_id"].to_csv("splits_val_ids.txt", index=False, header=False)
test_df["text_id"].to_csv("splits_test_ids.txt", index=False, header=False)