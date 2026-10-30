#from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import pandas as pd

#
# Let's use random forest or SVM
# Let's use classification report
#

# AAC, DPC, QSO
# CTDC, CTDT

neg_aac = pd.read_csv("neg_aac.csv")
neg_dpc = pd.read_csv("neg_dpc.csv")
neg_qso = pd.read_csv("neg_qso.csv")

neg_ctdc = pd.read_csv("neg_ctdc.csv")
neg_ctdt = pd.read_csv("neg_ctdt.csv")


pos_aac = pd.read_csv("pos_aac.csv")
pos_dpc = pd.read_csv("pos_dpc.csv")
pos_qso = pd.read_csv("pos_qso.csv")

pos_ctdc = pd.read_csv("pos_ctdc.csv")
pos_ctdt = pd.read_csv("pos_ctdt.csv")

# AAC: F1 on class 0 = zero, effectively
# DPC: F1 on class 0 = zero, effectively
# QSO: F1 on class 0 = zero, effectively
# CTDC: F1 on class 0 = zero, effectively
# CTDT: F1 on class 0 = zero, effectively
# All: F1 on class 0 = zero, effectively
def eval_svc_on(pos, neg):
	clf = SVC()

	x = pd.concat([pos, neg])
	y = [1]*len(pos) + [0]*len(neg)

	clf.fit(x, y)

	print(classification_report(y, clf.predict(x)))

# AAC: F1 on class 1 = 0.79
# DPC: F1 on class 1 = 0.21
# QSO: F1 on class 1 = 0.26
# CTDC: F1 on class 1 = 0.79
# CTDT: F1 on class 1 = 0.71
# All: F1 on class 1 = 0.79
def eval_svc_bal_on(pos, neg):
	clf = SVC()

	trunc_to = len(pos)*2

	x = pd.concat([pos, neg])
	y = [1]*len(pos) + [0]*len(neg)

	clf.fit(x[:trunc_to], y[:trunc_to])

	print(classification_report(y, clf.predict(x)))

# AAC: F1 on class 1 = 0.57
# DPC: F1 on class 1 = 0.57
# QSO: F1 on class 1 = 0.57
# CTDC: F1 on class 1 = 0.57
# CTDT: F1 on class 1 = 0.57
# All: F1 on class 1 = 0.57
def eval_svcw_on(pos, neg):
	clf = SVC()

	x = pd.concat([pos, neg])
	y = [1]*len(pos) + [0]*len(neg)

	w = [1]*len(pos) + [1/(len(neg)/len(pos))]*len(neg)

	clf.fit(x, y, sample_weight=w)

	print(classification_report(y, clf.predict(x)))

# AAC: F1 on class 1 = 0.57
def eval_balanced_moe_svc(pos, neg):
	x = pd.concat([pos, neg])
	y = [1]*len(pos) + [0]*len(neg)

	model = BalancedBaggingClassifier(
		estimator=SVC(),
		n_estimators=100,
		sampling_strategy="auto",
		bootstrap=True,
		bootstrap_features=False,
		random_state=42
	)

	model.fit(x, y)

	print(classification_report(y, model.predict(x)))

pos_all = pos_aac.merge(pos_dpc).merge(pos_qso).merge(pos_ctdc).merge(pos_ctdt)
neg_all = neg_aac.merge(neg_dpc).merge(neg_qso).merge(neg_ctdc).merge(neg_ctdt)

#eval_svc_on(pos_aac, neg_aac)
eval_svc_bal_on(pos_aac, neg_aac)
#eval_svcw_on(pos_aac, neg_aac)

#eval_svc_on(pos_dpc, neg_dpc)
#eval_svc_bal_on(pos_dpc, neg_dpc)
#eval_svcw_on(pos_dpc, neg_dpc)

#eval_svc_on(pos_qso, neg_qso)
#eval_svc_bal_on(pos_qso, neg_qso)
#eval_svcw_on(pos_qso, neg_qso)

#eval_svc_on(pos_ctdc, neg_ctdc)
#eval_svc_bal_on(pos_ctdc, neg_ctdc)
#eval_svcw_on(pos_ctdc, neg_ctdc)

#eval_svc_on(pos_ctdt, neg_ctdt)
#eval_svc_bal_on(pos_ctdt, neg_ctdt)
#eval_svcw_on(pos_ctdt, neg_ctdt)



#eval_svc_on(pos_all, neg_all)
#eval_svc_bal_on(pos_all, neg_all)
#eval_svcw_on(pos_all, neg_all)

# All: F1 on class 1 = 0.03
def eval_forest_on(pos, neg):
	clf = RandomForestClassifier(10, max_depth=2)

	x = pd.concat([pos, neg])
	y = [1]*len(pos) + [0]*len(neg)

	clf.fit(x, y)

	print(classification_report(y, clf.predict(x)))

# All: F1 on class 1 = 0.33
def eval_forest_bal_on(pos, neg):
	clf = RandomForestClassifier(10, max_depth=2)

	trunc_to = len(pos)*2

	x = pd.concat([pos, neg])
	y = [1]*len(pos) + [0]*len(neg)

	clf.fit(x[:trunc_to], y[:trunc_to])

	print(classification_report(y, clf.predict(x)))

# All: F1 on class 1 = 0.67
def eval_wforest_on(pos, neg):
	clf = RandomForestClassifier(10, max_depth=2)

	x = pd.concat([pos, neg])
	y = [1]*len(pos) + [0]*len(neg)

	w = [1]*len(pos) + [1/(len(neg)/len(pos))]*len(neg)

	clf.fit(x, y, sample_weight=w)

	print(classification_report(y, clf.predict(x)))

#eval_forest_on(pos_all, neg_all)
#eval_forest_bal_on(pos_all, neg_all)
#eval_wforest_on(pos_all, neg_all)

# All: F1 on class 1 = 1.00
def eval_mlp_on(pos, neg):
	clf = MLPClassifier((5, 5))

	x = pd.concat([pos, neg])
	y = [1]*len(pos) + [0]*len(neg)

	clf.fit(x, y)

	print(classification_report(y, clf.predict(x)))

# All: F1 on class 1 = 0.98
def eval_mlp_bal_on(pos, neg):
	clf = MLPClassifier((5, 5))

	trunc_to = len(pos)*2

	x = pd.concat([pos, neg])
	y = [1]*len(pos) + [0]*len(neg)

	clf.fit(x[:trunc_to], y[:trunc_to])

	print(classification_report(y, clf.predict(x)))

# All: F1 on class 1 = 0.94
def eval_mlpw_on(pos, neg):
	clf = MLPClassifier((5, 5))

	x = pd.concat([pos, neg])
	y = [1]*len(pos) + [0]*len(neg)

	w = [1]*len(pos) + [1/(len(neg)/len(pos))]*len(neg)

	clf.fit(x, y, sample_weight=w)

	print(classification_report(y, clf.predict(x)))

#eval_mlp_on(pos_all, neg_all)
#eval_mlp_bal_on(pos_all, neg_all)
#eval_mlpw_on(pos_all, neg_all)
