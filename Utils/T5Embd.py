import sys
from bio_embeddings.embed import ProtTransT5XLU50Embedder 
from os import write
import numpy as np


def embedBuilder(dataset, output):
    fin = open(dataset, "r")
    embedder = ProtTransT5XLU50Embedder()
    while True:
        line_PID = fin.readline().strip()[1:]
        line_Pseq = fin.readline().strip()
        #line_label = fin.readline() #.strip()
        if not line_Pseq:
            break
        if len(line_Pseq) < 1024:
            embedding = embedder.embed(line_Pseq)
            w = open("{}/{}.embd".format(output, line_PID), 'w')
            for cnt, aa in enumerate(line_Pseq):
                w.write(aa+':')
                w.write(' '.join([str(x) for x in embedding[cnt]]))
                w.write('\n')
    



def main():
    dataset = sys.argv[1]
    output = sys.argv[2]
    embedBuilder(dataset, output)

if __name__ == '__main__':
    main()
