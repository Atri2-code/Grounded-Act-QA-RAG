# Failure analysis (test split)

## Answerable questions whose gold section was not in the top 3

- q01: What exactly counts as a cheque?
  - gold: NIA s.6 (rank 8)
  - top 3: NIA s.129, NIA s.128, NIA s.89
- q02: Which three kinds of paper are negotiable instruments?
  - gold: NIA s.13 (rank 6)
  - top 3: NIA s.15, NIA s.56, NIA s.20
- q08: What marking makes a cheque crossed generally?
  - gold: NIA s.123 (rank 4)
  - top 3: NIA s.125, NIA s.126, NIA s.129
- q18: Can a cheque bounce case be settled between the parties without a trial verdict?
  - gold: NIA s.147 (rank None)
  - top 3: NIA s.143, NIA s.46, NIA s.43
- q24: Who is a holder in due course?
  - gold: NIA s.9 (rank 4)
  - top 3: NIA s.53, NIA s.36, NIA s.20
- q33: From what age must a pillion rider wear a helmet?
  - gold: MVA s.129 (rank 4)
  - top 3: MVA s.194C, MVA s.194D, MVA s.128
- q38: What is the punishment for driving while underage or without a licence?
  - gold: MVA s.181 (rank None)
  - top 3: MVA s.6, MVA s.182, MVA s.10
- q42: What happens to a rider's licence if caught without a helmet?
  - gold: MVA s.194D (rank None)
  - top 3: MVA s.7, MVA s.31, MVA s.10
- q43: What is the penalty for not giving way to an ambulance?
  - gold: MVA s.194E (rank None)
  - top 3: MVA s.194D, MVA s.177A, MVA s.210B
- q44: What is the penalty for driving without insurance?
  - gold: MVA s.196 (rank None)
  - top 3: MVA s.217, MVA s.194D, MVA s.156
- q45: If a minor commits a traffic offence who is held responsible?
  - gold: MVA s.199A (rank 4)
  - top 3: NIA s.141, MVA s.199, MVA s.210B
- q52: Is a vehicle allowed to run without registration?
  - gold: MVA s.39 (rank None)
  - top 3: MVA s.192, MVA s.161, MVA s.113

## Wrong answers given (after abstention)

- q16 (right section, key fact missing from the sentences picked): What is the cap on interim compensation a court can order in a cheque case?
  - cited: NIA s.143A
  - signals: dense 0.77, coverage 0.71
- q24 (wrong section): Who is a holder in due course?
  - cited: NIA s.53
  - signals: dense 0.89, coverage 1.00
- q52 (wrong section): Is a vehicle allowed to run without registration?
  - cited: MVA s.192
  - signals: dense 0.41, coverage 0.80
