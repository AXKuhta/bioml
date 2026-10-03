import re

from Bio.SeqIO.FastaIO import FastaIterator
from propy import PyPro

from tqdm import tqdm
import pandas as pd

#./T6SE_training_data/T6SE_Training_Neg_1112.fasta
#./T6SE_training_data/T6SE_Training_Pos_138.fasta

pos = []
neg = []

with open("T6SE_training_data/T6SE_Training_Pos_138.fasta") as f:
	for rec in FastaIterator(f):
		pos.append(rec)

with open("T6SE_training_data/T6SE_Training_Neg_1112.fasta") as f:
	for rec in FastaIterator(f):
		neg.append(rec)

# Feature extraction from amino acid sequence
def run_pipeline(examples):
	lst_aac = []
	lst_dpc = []
	lst_qso = []
	lst_ctdt = []
	lst_ctdc = []

	for item in tqdm(examples):
		desc = PyPro.GetProDes(item.seq)
		aac = desc.GetAAComp()
		dpc = desc.GetDPComp()
		qso = desc.GetQSO()

		lst_aac.append(aac)
		lst_dpc.append(dpc)
		lst_qso.append(qso)

		# Filter for class features +
		# Filter for transition features
		ctdc = dict()
		ctdt = dict()

		for k, v in desc.GetCTD().items():
			if re.match(r"^.+T\d+$", k):
				ctdt[k] = v
			elif re.match(r"^.+C\d+$", k):
				ctdc[k] = v
			else:
				pass

		lst_ctdt.append(ctdt)
		lst_ctdc.append(ctdc)

	df_aac = pd.DataFrame(lst_aac)
	df_dpc = pd.DataFrame(lst_dpc)
	df_qso = pd.DataFrame(lst_qso)
	df_ctdt = pd.DataFrame(lst_ctdt)
	df_ctdc = pd.DataFrame(lst_ctdc)

	return df_aac, df_dpc, df_qso, df_ctdc, df_ctdt

#
# Extract features from pos and neg
#

df_pos_aac, df_pos_dpc, df_pos_qso, df_pos_ctdc, df_pos_ctdt = run_pipeline(pos)
df_neg_aac, df_neg_dpc, df_neg_qso, df_neg_ctdc, df_neg_ctdt = run_pipeline(neg)

#
# Save
#

df_pos_aac.to_csv("pos_aac.csv")
df_pos_dpc.to_csv("pos_dpc.csv")
df_pos_qso.to_csv("pos_qso.csv")
df_pos_ctdc.to_csv("pos_ctdc.csv")
df_pos_ctdt.to_csv("pos_ctdt.csv")

df_neg_aac.to_csv("neg_aac.csv")
df_neg_dpc.to_csv("neg_dpc.csv")
df_neg_qso.to_csv("neg_qso.csv")
df_neg_ctdc.to_csv("neg_ctdc.csv")
df_neg_ctdt.to_csv("neg_ctdt.csv")
