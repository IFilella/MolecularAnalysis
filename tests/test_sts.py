from MolecularAnalysis.analysis import sts
import matplotlib.pyplot as plt

data=sts.loadSchrodingerCSV('../data/test.csv')

plt.figure()
sts.plotROC(data=data, RealClass='Value', PredClass='glide gscore', RealThrs=1e-5)
plt.savefig('testout/ROC.png')

plt.figure()
sts.plotMultConfusionPlots(data=data, RealClass='Value', PredClass='glide gscore',
                           RealThrs=1e-5, PredThrs=-5.5, disc='Job Name')
plt.savefig('testout/MultConfusion.png')

plt.figure()
sts.plotROC(data=data, RealClass='Value', PredClass='glide gscore', RealThrs=1e-5,
            whole_db=False, n_iter=15)
plt.savefig('testout/ROCbalanced.png')
