import sys
import os

def fastaBuilder(dataset, output):
    fin = open(dataset, "r")
    while True:
        line_PID = fin.readline().strip()[1:]
        line_Pseq = fin.readline().strip()
        print(line_Pseq, len(line_Pseq))
        print(line_PID, len(line_PID))
        #line_label = fin.readline().strip()
        if not line_Pseq:
            break
        if len(line_Pseq) < 1024 and len(line_Pseq) > 5:
            print(output, line_PID)
            os.system('mkdir {}/{}'.format(output, line_PID))
            w = open('{}/{}.txt'.format(output, line_PID), 'w')
            w.write('>'+line_PID+'\n')
            w.write(line_Pseq+'\n')
        else:
            print(line_PID)


def main():
    dataset = sys.argv[1]
    output = sys.argv[2]
    print(dataset,output)
    fastaBuilder(dataset, output)

if __name__ == '__main__':
    main()
