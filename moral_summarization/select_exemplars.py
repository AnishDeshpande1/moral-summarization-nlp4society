import os
import numpy as np
from sentence_transformers import SentenceTransformer
from moral_summarization.data_utils import (
    EMONA_dataset_path,
    EMONA_datasets,
    article_folders,
    load_annotations,
    get_moral_annotations,
    load_article,
    get_articles_in_test_set
)
from moral_summarization.utils import moral_labels

# Helper function to compute MFT cosine similarity
def mft_cosine_similarity(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return np.dot(v1, v2) / (n1 * n2)

# Helper function to map articles to their ideology
def get_article_ideology(name, dataset):
    """
    Extract political ideology stance:
    Left: -1, Center: 0, Right: 1
    
    AllSides mapping:
      Filename: allsides_<topic>_<ideology>_<id>.txt
      ideology is the second-to-last element (split by '_'):
      'l' -> Left, 'c' -> Center, 'r' -> Right
      
    BASIL mapping:
      Filename: basil_<id>_<source>.txt
      source is the last element (split by '_'):
      'hpo' -> Left, 'nyt' -> Center, 'fox' -> Right
      
    MPQA is excluded from the candidate training pool since it has no political stance.
    """
    parts = name.split('_')
    if dataset == 'allsides':
        ideology_char = parts[-2]
        if ideology_char == 'l':
            return -1
        elif ideology_char == 'c':
            return 0
        elif ideology_char == 'r':
            return 1
    elif dataset == 'basil':
        source = parts[-1]
        if source == 'hpo':
            return -1
        elif source == 'nyt':
            return 0
        elif source == 'fox':
            return 1
    return None

class ExemplarSelector:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ExemplarSelector, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        if self._initialized:
            return
        
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
        # Load and precompute candidate pools
        self.test_articles = set(get_articles_in_test_set())
        self.candidates = []
        
        self._load_candidates()
        self._precompute_candidate_embeddings()
        self._initialized = True

    def _load_candidates(self):
        """
        Scan EMONA datasets to build candidate training exemplar pools.
        Exclude test set articles and MPQA articles.
        """
        # Only use allsides and basil for exemplars
        for dataset in ['allsides', 'basil']:
            dataset_folder = os.path.join(EMONA_dataset_path, article_folders[dataset])
            if not os.path.exists(dataset_folder):
                continue
            
            for file_name in os.listdir(dataset_folder):
                if not file_name.endswith('.txt'):
                    continue
                
                article_name = os.path.splitext(file_name)[0]
                if article_name in self.test_articles:
                    continue
                
                ideology = get_article_ideology(article_name, dataset)
                if ideology is None:
                    continue  # Ignore if ideology mapping is not resolved
                
                try:
                    text = load_article(article_name, dataset)
                    annotations = load_annotations(article_name, dataset)
                except Exception as e:
                    print(f"Warning: Failed to load candidate {article_name}: {e}")
                    continue
                
                # Compute 10-dimensional MFT frequency vector
                mft_vector = np.zeros(10)
                moral_annotations = get_moral_annotations(annotations)
                for ann in moral_annotations:
                    label = ann['label']
                    if label in moral_labels:
                        idx = moral_labels.index(label)
                        mft_vector[idx] += 1
                
                self.candidates.append({
                    'name': article_name,
                    'dataset': dataset,
                    'text': text,
                    'ideology': ideology,
                    'mft_vector': mft_vector
                })

    def _precompute_candidate_embeddings(self):
        """
        Precompute and cache dense semantic embeddings for all candidates.
        """
        if not self.candidates:
            self.candidate_embeddings = np.array([])
            return
            
        texts = [c['text'] for c in self.candidates]
        self.candidate_embeddings = self.model.encode(texts, show_progress_bar=False)
        
        # Categorize candidates into Left, Center, Right buckets with their indices
        self.left_indices = []
        self.center_indices = []
        self.right_indices = []
        
        for idx, c in enumerate(self.candidates):
            if c['ideology'] == -1:
                self.left_indices.append(idx)
            elif c['ideology'] == 0:
                self.center_indices.append(idx)
            elif c['ideology'] == 1:
                self.right_indices.append(idx)

    def select_exemplars(self, target_text, target_article_name=None, alpha=0.5, beta=0.5, top_n=10):
        """
        Dynamically select the optimal 3-shot exemplars (E* = {e_L, e_C, e_R}) for a target article T.
        
        Formula:
          E* = ArgMax_E [ Alpha * Sum_{e in E} Sim(e, T) + Beta * (3 - Sim(e_L, e_C) - Sim(e_L, e_R) - Sim(e_C, e_R)) ]
          Subject to one exemplar from Left, Center, Right.
        """
        if not self.candidates:
            raise ValueError("No candidates loaded in exemplar pool.")
            
        # Encode target text
        target_embedding = self.model.encode([target_text], show_progress_bar=False)[0]
        target_norm = np.linalg.norm(target_embedding)
        
        if target_norm == 0:
            target_norm = 1e-9
            
        # Compute semantic cosine similarity to all candidates
        similarities = []
        for idx in range(len(self.candidates)):
            cand_emb = self.candidate_embeddings[idx]
            cand_norm = np.linalg.norm(cand_emb)
            if cand_norm == 0:
                cand_norm = 1e-9
            sim = np.dot(cand_emb, target_embedding) / (cand_norm * target_norm)
            similarities.append(sim)
            
        # Filter target article itself if present in candidates (safety check)
        if target_article_name is not None:
            for idx, c in enumerate(self.candidates):
                if c['name'] == target_article_name:
                    similarities[idx] = -1.0  # Force it out of selection
                    
        # Get top-N candidates for each ideology bucket
        def get_top_n_for_bucket(indices):
            bucket_similarities = [(idx, similarities[idx]) for idx in indices]
            # Sort by similarity descending
            bucket_similarities.sort(key=lambda x: x[1], reverse=True)
            return [idx for idx, _ in bucket_similarities[:top_n]]
            
        top_left = get_top_n_for_bucket(self.left_indices)
        top_center = get_top_n_for_bucket(self.center_indices)
        top_right = get_top_n_for_bucket(self.right_indices)
        
        # Combinatorial search to find the optimal triplet
        best_score = -float('inf')
        best_triplet = None
        
        for l_idx in top_left:
            l_cand = self.candidates[l_idx]
            l_sim = similarities[l_idx]
            
            for c_idx in top_center:
                c_cand = self.candidates[c_idx]
                c_sim = similarities[c_idx]
                
                for r_idx in top_right:
                    r_cand = self.candidates[r_idx]
                    r_sim = similarities[r_idx]
                    
                    # Compute MFT cosine similarities for Diversity
                    sim_lc = mft_cosine_similarity(l_cand['mft_vector'], c_cand['mft_vector'])
                    sim_lr = mft_cosine_similarity(l_cand['mft_vector'], r_cand['mft_vector'])
                    sim_cr = mft_cosine_similarity(c_cand['mft_vector'], r_cand['mft_vector'])
                    
                    # MFT Diversity score
                    diversity_score = 3.0 - (sim_lc + sim_lr + sim_cr)
                    
                    # Total semantic relevance score
                    relevance_score = l_sim + c_sim + r_sim
                    
                    # Objective function
                    score = alpha * relevance_score + beta * diversity_score
                    
                    if score > best_score:
                        best_score = score
                        best_triplet = (l_cand, c_cand, r_cand)
                        
        return best_triplet
