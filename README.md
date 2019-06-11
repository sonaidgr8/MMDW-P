# MMDW-Python
    Python Implementation of Max-Margin DeepWalk :- https://github.com/thunlp/MMDW
    Paper :- "Max-Margin DeepWalk: Discriminative Learning of Network Representation" https://www.ijcai.org/Proceedings/16/Papers/547.pdf

## Input:- 
    Input is organized as follows -
    Datasets/
        |_ _ _ <Dataset-name>
                    |_ _ _ <Dataset-name.mat>
                    |_ _ _ <Stats.txt>
                    |_ _ _ <Percentage of train-test split>
                                        |_ _ _ <Fold-No>
                                                    |_ _ _ test_ids.npy
                                                    |_ _ _ train_ids.npy
                                                    |_ _ _ val_ids.npy
## Usage:-
    python main_algo.py --DATA_DIR <Dataset name> --ALPHA_BIAS <Alpha-bias for biased RW> --ALPHA <Proximity matrix weight> --LAMBDA <L2 regularization weight> --L_COMPONENTS <Dimension of projection>                
    python main_algo.py --DATA_DIR cora --ALPHA_BIAS -2 --ALPHA 1.0 --LAMBDA 1.0 --L_COMPONENTS 16
    
## Output:-
    * The generated node and label embeddings are saved in Emb/ folder as <emb_dataset_U<Fold-No>>.npy and <emb_dataset_Q<Fold-No>>.npy of dimension (#Nodes x L_COMPONENTS) and (#Labels x L_COMPONENTS) respectively.   
    * The Node Classification evaluation results are stored in Results/ folder as <dataset>_best_params_node_classification.txt
