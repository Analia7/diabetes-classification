N_TRAIN=-1
N_TEST=-1
N_JOBS=10

CRIT=('gini' 'entropy')
MIN=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9) # This is used for min_samples_split, min_samples_leaf, min_weight_fractional and min_impurity_decrease
MAXFEAT=('0.1' '0.2' '0.3' '0.4' '0.5' '0.6' '0.7' '0.8' '0.9' '1.0' 'sqrt' 'log2')
CLWEIGHT=('None' 'balanced')
SPLIT=('best' 'random')

for CR in ${CRIT[@]} # criterion
do
    for SSP in ${MIN[@]} # min_samples_split
    do
        for SLF in ${MIN[@]} # min_samples_leaf
        do
            for WFRC in ${MIN[@]} # min_weight_fractional
            do
                for MFT in ${MAXFEAT[@]} # max_features
                do
                    for IMP in ${MIN[@]} # min_impurity_decrease
                    do
                        for CW in ${CLWEIGHT[@]} # class_weight
                        do
                            for SP in ${SPLIT[@]} # splitter
                            do
                                echo $ALG, $HP
                                python ./run_test.py --data ../data --tr_it $N_TRAIN --te_it $N_TEST -o ./results/test_run1/ --n_jobs $N_JOBS --alg nc --thr $HP --pt $ALG
                            done
                        done
                    done
                done
            done
        done
    done
done


ALGOS=('anm' 'bivariate_fit' 'cds' 'gnn' 'igci' 'reci')
HPS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for ALG in ${ALGOS[@]}
do
    for HP in ${HPS[@]}
    do
        echo $ALG, $HP
        python ./run_test.py --data ../data --tr_it $N_TRAIN --te_it $N_TEST -o ./results/test_run1/ --n_jobs $N_JOBS --alg nc --thr $HP --pt $ALG
    done
done