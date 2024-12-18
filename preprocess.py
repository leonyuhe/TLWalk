

from __future__ import division

import os
import random

import networkx as nx
import numpy as np

from evalne.evaluation.evaluator import LPEvaluator
from evalne.evaluation.score import Scoresheet
from evalne.evaluation.split import LPEvalSplit
from evalne.utils import preprocess as pp
from evalne.utils import split_train_test as stt

# NOTE: The example `as is`, only evaluates baseline methods. To evaluate the OpenNE methods, PRUNE and Metapath2vec
# these must be first installed. Then the correct paths must be set in the commands_other variable.
# Finally, the following parameter can be set to True.
run_other_methods = False


def main():
    # Initialize some parameters
    inpath = list()
    nw_names = ['aves-weaver-social', 'bio-CE-LC','bio-DM-LC',
                'bn-cat-mixed-species_brain_1',
                'soc-wiki-Vote', 'fb-pages-food','soc-hamsterster','ego-Facebook',
                 'bio-CE-HT', 'bio-celegans-dir','bio-WormNet-v3']   # Stores the names of the networks evaluated
    inpath.append("./input/aves-weaver-social.edges")
    inpath.append("./input/bio-CE-LC.edges")
    inpath.append("./input/bio-DM-LC.edges")
    inpath.append("./input/bn-cat-mixed-species_brain_1.edges")
    inpath.append("./input/soc-wiki-Vote.edges")
    inpath.append("./input/fb-pages-food.edges")
    inpath.append("./input/soc-hamsterster.edges")
    inpath.append("./input/ego-Facebook.edges")
    inpath.append("./input/bio-CE-HT.edges")
    inpath.append("./input/bio-celegans-dir.edges")
    inpath.append("./input/bio-WormNet-v3.edges")
    outpath = "./output/"

    if not os.path.exists(outpath):
        os.makedirs(outpath)
    directed = False		        # indicates if the graphs are directed or undirected
    delimiters = (' ', ' ', ' ', ' ', ' ', ',', ' ', ' ', ' ', ' ', ' ')		# indicates the delimiter in the original graph
    repeats = 2		                # number of time the experiment will be repeated

    # Create a scoresheet to store the results
    scoresheet = Scoresheet(tr_te='test')

    for i in range(len(inpath)):

        # Create folders for the evaluation results (one per input network)
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        # Load and preprocess the graph
        G = preprocess(inpath[i], nw_names[i], outpath, delimiters[i], directed)
        pp.get_stats(G)

        # Alternatively, train/test splits can be computed one at a time
        train_E, test_E = stt.split_train_test(G=G, train_frac=0.80)
        train_E_false, test_E_false = stt.generate_false_edges_cwa(G, train_E=train_E, test_E=test_E,
                                                                   num_fe_train=None,
                                                                   num_fe_test=None)
        stt.store_train_test_splits(os.path.join(outpath, "lp_train_test_splits", nw_names[i]),
                                                train_E=train_E, train_E_false=train_E_false, test_E=test_E,
                                                test_E_false=test_E_false, split_id=0)
        TG = nx.Graph()
        TG.add_edges_from(train_E)
        nx.write_edgelist(TG, outpath+nw_names[i]+".train.edgelist", delimiter=",", data=False)


            # # Evaluate baselines
            # eval_baselines(nee, directed, scoresheet)
            #
            # # Evaluate other NE methods
            # if run_other_methods:
            #     eval_other(nee, scoresheet)

    # print("\nEvaluation results:")
    # print("-------------------")
    #
    # # Print results averaged over exp repeats
    # scoresheet.print_tabular(metric='auroc')
    #
    # # Write results averaged over exp repeats to a single file
    # scoresheet.write_tabular(filename=os.path.join(outpath, 'eval_output.txt'), metric='auroc')
    #
    # # Store the Scoresheet object for later analysis
    # scoresheet.write_pickle(os.path.join(outpath, 'eval.pkl'))
    #
    # print("Evaluation results are also stored in a folder named `output` in the current directory.")
    # print("End of evaluation")


def preprocess(inpath, name, outpath, delimiter, directed):
    """
    Graph preprocessing routine.
    """
    print('Preprocessing graph...')

    # Load a graph
    G = pp.load_graph(inpath, delimiter=delimiter, comments='#', directed=directed)

    # Preprocess the graph
    G, ids = pp.prep_graph(G, relabel=True, del_self_loops=True)

    # Store preprocessed graph to a file
    pp.save_graph(G, output_path=outpath + name+ ".prep_graph.edgelist", delimiter=',', write_stats=True)

    # Return the preprocessed graph
    return G





if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()