with open('email-Eu-core.txt') as fin, open('email-core.dat','w') as fout:
    for line in fin:
        u,v = line.split()
        fout.write(f"{u} {v} 1.0\n")