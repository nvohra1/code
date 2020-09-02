
inputfile = "DNA_NM_207618.2.txt"
f = open(inputfile, "r")
seqRaw = f.read()
print("Q3: ", seqRaw[40:50])
seq1 = seqRaw.replace("\n","")
print(seq1)
seq = seq1.replace("\r","")
print(seq)

# lookup from dictionary
# check seq2 length is divisable by 3
    # loop over the sequence
        # extract a single codon(3Char)
        # look up the codon and store their resut.

print(len(seqRaw))
print(len(seq))
print(len(seqRaw) - len(seq))

def translate(seq):
    """Translate a string containing a nucleotide sequence into a string
    containing the corrosponding sequence of amino acids. Nucleotides are
    translated in triplets using the table dictionary; each amino acid
    is encoded with a string of length 1."""

    table = {
        'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
        'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
        'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
        'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
        'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
        'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
        'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
        'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
        'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
        'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
        'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
        'TAC': 'Y', 'TAT': 'Y', 'TAA': '_', 'TAG': '_',
        'TGC': 'C', 'TGT': 'C', 'TGA': '_', 'TGG': 'W',
    }

    protein = ""
    if len(seq) % 3 == 0:
        # print(True)
        for i in range(0, len(seq), 3):
            codon = seq[i : i+3]
            protein += table[codon]
    else:
        print("translate Failed")
    return protein
# docstring

ret = translate(seq)
#ret = translate('ATAATCGCC')
print(ret)
print(help(translate))

##########################################################################

def read_seq(inputfile):
    """Reads the file and return the input sequence with special characters removed"""
    with open(inputfile, "r") as f:
        seqRaw = f.read()
    seq1 = seqRaw.replace("\n", "")
    seq = seq1.replace("\r", "")
    return seq


inputfile_dna = "dna.txt"
inputfile_protein = "protein.txt"

seq_Dna = read_seq(inputfile_dna)
seq_ptn = read_seq(inputfile_protein)

print(seq_Dna)
print(len(seq_Dna), len(seq_Dna)%3)
print(seq_ptn)
print(len(seq_ptn), len(seq_ptn)%3)



# https://www.ncbi.nlm.nih.gov/nuccore/NM_207618.2
# CDS             21..938   which mean python sequence [20:938
# But Last codon is stop codon so it is not included in protein sequence.
# so ignore last codon now python sequence is [20;935]
op_dna = translate(seq_Dna[20:935])
print("if translated matches give protein seq: ", seq_ptn == op_dna)
print(op_dna)
print(len(op_dna))

# Another way of doing the same
op_dna = translate(seq_Dna[20:938])[:-1]
print("if translated matches give protein seq: ", seq_ptn == op_dna)
print(op_dna)
print(len(op_dna))