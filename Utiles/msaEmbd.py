import sys
from os import write
from biotransformers import BioTransformers
import numpy as np




def embedBuilder(dataset, a3m, output):
    fin = open(dataset, "r")
    bio_trans = BioTransformers("esm_msa1_t12_100M_UR50S",num_gpus=0)
    while True:
        line_PID = fin.readline().strip()[1:]
        line_Pseq = fin.readline().strip()
        #line_label = fin.readline().strip()
        if not line_Pseq:
            break

        if len(line_Pseq) < 1024:
            num_lines = sum(1 for line in open('{}/{}/{}.a3m'.format(a3m, line_PID, line_PID)))//2
            if num_lines < 64:
                print("low {}-{}".format(line_PID, num_lines))
            msa_file = '{}/{}/'.format(a3m, line_PID) #open('a3m/{}.fasta.a3m'.format(line_PID))
            msa_embeddings = bio_trans.compute_embeddings(sequences=msa_file, pool_mode=("cls","mean","full"), n_seqs_msa=min(64,num_lines))
            embd = np.array(msa_embeddings['full'])[0,0]
            w = open("{}/{}.embd".format(output, line_PID), 'w')
            for cnt, aa in enumerate(line_Pseq):
                w.write(aa+':')
                w.write(' '.join([str(x) for x in embd[cnt]]))
                w.write('\n')
            
def main():
    dataset = sys.argv[1]
    a3m = sys.argv[2]
    output = sys.argv[3]
    embedBuilder(dataset, a3m, output)

if __name__ == '__main__':
    main()
