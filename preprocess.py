from utils.TRNDataLoader import TRNDataLoader

loader = TRNDataLoader("data/raw")
i = 0
for sent in loader.generate():
    i += 1

print(i)
