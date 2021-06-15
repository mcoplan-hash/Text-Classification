with open("/mnt/data/mcoplan/Text-Classification/data/glove/BioWordVec_PubMed_MIMICIII_d200.bin", "rb") as file:
    data = file.read(8)

with open("/mnt/data/mcoplan/Text-Classification/data/glove/embedding.txt", "w") as f:
    f.write(" ".join(map(str, data)))
    f.write("\n")
