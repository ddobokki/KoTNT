# import os

# import pandas as pd
# from datasets import Dataset


# def get_trn_datasets(csv_path:os.PathLike) -> Dataset:
#     df = pd.read_csv(csv_path)
#     num_col = df['num_col'].tolist()
#     sen_col = df['sen_col'].tolist()

#     dataset = Dataset.from_dict({
#         "num_col":num_col,
#         "sen_col":sen_col
#     })
#     return dataset
