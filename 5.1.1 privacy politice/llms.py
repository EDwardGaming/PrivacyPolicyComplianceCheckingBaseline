import json
import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Import base classifier and config
from kfc import LlamaBatchClassifier 
from Config import Config

class LlamaIndexRAGClassifier(LlamaBatchClassifier):
    """
    RAG Dataset Builder.
    Uses Sliding Window approach to retrieve context (neighbors).
    Uses LLM API (ChatGPT-4) to generate Document Summaries.
    Outputs JSONL files for fine-tuning a 7B model elsewhere.
    """
    def __init__(self):
        super().__init__()
        self.summaries: Dict[str, str] = {}
        # Cache for document sentences to avoid repeated grouping
        self.doc_sentences: Dict[str, List[str]] = {}

    def _generate_document_summary(self, document_text: str) -> str:
        """
        Use the LLM API to generate a concise summary of the document.
        """
        system_prompt = "You are a professional legal document analyst."
        prompt = (
            "Please provide a concise summary of the following Privacy Policy document. "
            "Focus on data collection practices, user rights, and data processing purposes. "
            "Keep the summary within 200 words.\n\n"
            f"Document Content (Excerpt):\n{document_text}..." # Truncate to avoid context limit
        )
        
        try:
            # Reuse the base class's API call logic
            response = self._call_llm_api(prompt, system_prompt=system_prompt)
            return response if response else "Summary generation failed."
        except Exception as e:
            print(f"Summary generation error: {e}")
            return "Summary not available."

    def prepare_data(self, df: pd.DataFrame):
        """
        Prepare Data:
        1. Generate Summaries for each document.
        2. Cache sentences for window retrieval.
        """
        unique_files = df['filename'].unique()
        print(f"\n>>> Preparing RAG Data (Summaries & Contexts) for {len(unique_files)} documents...")
        
        for filename in tqdm(unique_files, desc="Building RAG Dataset"):
            # Combine all sentences to form the document text
            file_df = df[df['filename'] == filename]
            sentences = file_df['sentence'].tolist()
            full_text = " ".join(sentences)
            
            # 1. Generate Summary (if not cached)
            if filename not in self.summaries:
                self.summaries[filename] = self._generate_document_summary(full_text)
                
            # 2. Cache sentences for window retrieval
            if filename not in self.doc_sentences:
                self.doc_sentences[filename] = sentences

    def _get_window_context(self, filename: str, target_sentence: str, window_size: int = 3) -> str:
        """
        Retrieve surrounding sentences as context.
        """
        sentences = self.doc_sentences.get(filename, [])
        if not sentences:
            return ""
            
        try:
            # Find index of target sentence
            idx = sentences.index(target_sentence)
            start = max(0, idx - window_size)
            end = min(len(sentences), idx + window_size + 1)
            context_slice = sentences[start:end]
            return "\n".join(context_slice)
        except ValueError:
            return target_sentence

    def _build_rag_prompt(self, target_sentence: str, summary: str, context_text: str) -> str:
        """
        Construct the 3-part prompt:
        1. Summary
        2. Context
        3. Target Sentence
        """
        prompt = (
            "The information that assists you in making your judgment consists of the following three parts: "
            "Part 1: Document Summary, Part 2: Context Information (Retrieved), and Part 3: Target Sentence.\n\n\n"
            f"### Part 1: Document Summary\n{summary}\n\n"
            f"### Part 2: Context Information (Retrieved)\n{context_text}\n\n"
            f"### Part 3: Target Sentence\n{target_sentence}\n\n"
            "### Instruction\n"
            "Based on the summary, context, and the target sentence, classify the target sentence into one of the 10 categories (0-10). "
            "Output ONLY a single integer representing the category.\n\n"
            "Example Input(Target Sentence in Part 3:):\n"
            "We collect your email.\n"
            "Example Output:\n"
            "1"
        )
        return prompt

    def _balance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance the dataset:
        - Downsample Class 0 (Other) to ~2500.
        - Upsample Minority Classes (< 400) by duplication.
        - Keep others as is.
        """
        print(f"\n[Data Balancing] Original distribution:\n{df['label'].value_counts().sort_index()}")
        
        dfs = []
        for label in sorted(df['label'].unique()):
            sub_df = df[df['label'] == label]
            count = len(sub_df)
            
            if label == 0:
                # Downsample Class 0
                target_n = 2500
                if count > target_n:
                    sub_df = sub_df.sample(n=target_n, random_state=Config.RANDOM_STATE)
            elif count < 400:
                # Upsample Minority to reach at least 400
                # Duplicate until we reach the threshold
                while len(sub_df) < 400:
                    sub_df = pd.concat([sub_df, sub_df])
                # Optional: Shuffle the duplicates
                sub_df = sub_df.sample(frac=1, random_state=Config.RANDOM_STATE)
            
            dfs.append(sub_df)
        
        balanced_df = pd.concat(dfs).sample(frac=1, random_state=Config.RANDOM_STATE).reset_index(drop=True)
        print(f"[Data Balancing] Final distribution:\n{balanced_df['label'].value_counts().sort_index()}")
        print(f"[Data Balancing] Total samples: {len(balanced_df)}")
        return balanced_df

    def generate_dataset(self, X, y, filenames, output_file, balance=False, return_df=False):
        """
        Generate LLM fine-tuning dataset (JSONL format) with Instruction/Input/Response.
        """
        # Create DataFrame
        df = pd.DataFrame({'sentence': X, 'label': y, 'filename': filenames})
        
        # 1. Prepare Data (Summaries & Contexts) using FULL dataset
        self.prepare_data(df)
        
        # 2. Balance Data if requested (Only for Training set)
        if balance:
            df = self._balance_data(df)
        else:
            # Shuffle to mix sentences from different documents for Val/Test sets
            df = df.sample(frac=1, random_state=Config.RANDOM_STATE).reset_index(drop=True)
        
        print(f"\n>>> Generating Dataset: {output_file} ({len(df)} samples)...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating JSONL"):
                filename = row['filename']
                sentence = row['sentence']
                label = row['label']
                
                # Get Summary and Context
                summary = self.summaries.get(filename, "")
                context_text = self._get_window_context(filename, sentence, window_size=3)
                
                # Build User Prompt (The 4-part prompt)
                user_prompt = self._build_rag_prompt(sentence, summary, context_text)
                
                # Construct JSON Entry
                entry = {
                    "Instruction": f"{self.base_prompt}\n\n{user_prompt}",
                    "Input": sentence,
                    "Response": str(label)
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        print(f"Successfully saved to {output_file}")

        if return_df:
            return df

    def generate_ablation_dataset(self, X, y, filenames, output_file, source_df=None):
        """
        Generate ablation dataset (No Summary, No Context).
        If source_df is provided, use it (to match balanced training data).
        """
        # Create DataFrame if not provided
        if source_df is None:
            df = pd.DataFrame({'sentence': X, 'label': y})
            # Shuffle to mix sentences from different documents
            df = df.sample(frac=1, random_state=Config.RANDOM_STATE).reset_index(drop=True)
        else:
            df = source_df
        
        print(f"\n>>> Generating Ablation Dataset: {output_file} ({len(df)} samples)...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating JSONL (Ablation)"):
                sentence = row['sentence']
                label = row['label']
                
                # Build User Prompt (Target Sentence only)
                user_prompt = (
                    f"### Target Sentence\n{sentence}\n\n"
                    "### Instruction\n"
                    "Classify the target sentence into one of the 10 categories (0-10). "
                    "Output ONLY a single integer representing the category.\n\n"
                    "Example Input:\n"
                    "We collect your email.\n"
                    "Example Output:\n"
                    "1"
                )
                
                # Construct JSON Entry
                entry = {
                    "Instruction": f"{self.base_prompt}\n\n{user_prompt}",
                    "Input": sentence,
                    "Response": str(label)
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        print(f"Successfully saved to {output_file}")

# ==================== 整合进 run_experiment ====================
def run_experiment_v2(X_test, y_test, files_test, train_data=None, val_data=None):
    """
    Main entry point for RAG Dataset Construction.
    Generates:
    - rag_train.jsonl
    - rag_val.jsonl
    - rag_test.jsonl
    - ablation_train.jsonl
    - ablation_val.jsonl
    - ablation_test.jsonl
    """
    print("="*60)
    print("Running LlamaIndex RAG & Ablation Dataset Construction")
    print("="*60)
    
    rag_classifier = LlamaIndexRAGClassifier()
    
    # 1. Generate Training Data
    if train_data:
        X_train, y_train, files_train = train_data
        # Balance only the training data
        balanced_train_df = rag_classifier.generate_dataset(X_train, y_train, files_train, "rag_train.jsonl", balance=True, return_df=True)
        rag_classifier.generate_ablation_dataset(X_train, y_train, files_train, "ablation_train.jsonl", source_df=balanced_train_df)
    
    # 2. Generate Validation Data
    if val_data:
        X_val, y_val, files_val = val_data
        rag_classifier.generate_dataset(X_val, y_val, files_val, "rag_val.jsonl")
        rag_classifier.generate_ablation_dataset(X_val, y_val, files_val, "ablation_val.jsonl")
    
    # 3. Generate Test Data (for later inference)
    rag_classifier.generate_dataset(X_test, y_test, files_test, "rag_test.jsonl")
    rag_classifier.generate_ablation_dataset(X_test, y_test, files_test, "ablation_test.jsonl")
    
    print("\n" + "="*60)
    print("DATASET GENERATION COMPLETE.")
    print("Generated Innovation Datasets: rag_train.jsonl, rag_val.jsonl, rag_test.jsonl")
    print("Generated Ablation Datasets: ablation_train.jsonl, ablation_val.jsonl, ablation_test.jsonl")
    print("Please download these files to your GPU platform for fine-tuning.")
    print("="*60 + "\n")
    
    # Return dummy predictions to satisfy the calling signature in GdprComplianceExperiment.py
    # We return 0s so the pipeline finishes without error.
    return np.zeros(len(X_test), dtype=int), np.arange(len(X_test))

if __name__ == "__main__":
    # Standalone execution for Dataset Generation
    print(">>> Starting Standalone RAG Dataset Generation...")
    
    # 1. Load Data
    if not Config.DATA_PATH:
        raise ValueError("Config.DATA_PATH is not set")
        
    print(f"Loading data from {Config.DATA_PATH}")
    df = pd.read_csv(Config.DATA_PATH, sep='\t', encoding='utf-8')
    
    # 2. Split Data (Same logic as GdprComplianceExperiment.py)
    unique_files = df['filename'].unique()
    
    # [DEBUG] Limit to a small number of documents for testing
    DEBUG_DOC_LIMIT = None  # 修改此处：10 表示只跑10篇文档；设为 None 或 0 则跑全量数据
    if DEBUG_DOC_LIMIT and len(unique_files) > DEBUG_DOC_LIMIT:
        print(f"\n⚠️ DEBUG MODE: Limiting to {DEBUG_DOC_LIMIT} documents for testing...")
        unique_files = unique_files[:DEBUG_DOC_LIMIT]
        # Filter dataframe to only these files to ensure consistency
        df = df[df['filename'].isin(unique_files)]
        print(f"Filtered dataframe size: {len(df)} sentences")

    # Handle small dataset sizes for debugging
    if len(unique_files) < 3:
        print("⚠️ Too few documents for splitting. Using full set for Train/Val/Test.")
        train_files_unique = unique_files
        val_files_unique = unique_files
        test_files_unique = unique_files
    else:
        # Split Train+Val vs Test
        train_val_files, test_files_unique = train_test_split(
            unique_files, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
        )
        
        # Split Train vs Val
        train_files_unique, val_files_unique = train_test_split(
            train_val_files, test_size=Config.TEST_SIZE/(1-Config.TEST_SIZE), # Approx 10% of total
            random_state=Config.RANDOM_STATE
        )
    
    def get_data_by_files(source_df, target_files):
        mask = source_df['filename'].isin(target_files)
        return (
            source_df[mask]['sentence'].values, 
            source_df[mask]['label'].values, 
            source_df[mask]['filename'].values
        )

    X_train, y_train, files_train = get_data_by_files(df, train_files_unique)
    X_val, y_val, files_val = get_data_by_files(df, val_files_unique)
    X_test, y_test, files_test = get_data_by_files(df, test_files_unique)
    
    # 3. Run Generation
    run_experiment_v2(
        X_test, y_test, files_test,
        train_data=(X_train, y_train, files_train),
        val_data=(X_val, y_val, files_val)
    )
    print(">>> Standalone Generation Complete.")
