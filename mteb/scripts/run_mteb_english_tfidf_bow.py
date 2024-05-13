import logging
from mteb import MTEB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer
import imodelsx.auglinear.embed
from spacy.lang.en import English
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


class TfidfTransformer(SentenceTransformer):
    """Class that uses TF-IDF vectorization to convert text into a matrix of TF-IDF features."""

    def __init__(self, ngram_range=(1, 1)):
        """
        Params
        ------
        ngram_range (tuple): The lower and upper boundary of the range of n-values for different n-grams to be extracted.
        """
        super().__init__(model_name_or_path=None)
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range)
        self.is_fitted = False

    def fit(self, texts):
        self.vectorizer.fit(texts)
        self.is_fitted = True

    def encode(self, texts, **kwargs):
        if not self.is_fitted:
            self.fit(texts)
        return self.vectorizer.transform(texts).toarray()

    def transform(self, texts):
        return self.encode(texts)


class BowTransformer(SentenceTransformer):
    """Class that uses Bag of Words (BoW) vectorization to convert text into a matrix of token counts."""

    def __init__(self, ngram_range=(1, 1)):
        """
        Params
        ------
        ngram_range (tuple): The lower and upper boundary of the range of n-values for different n-grams to be extracted.
        """
        super().__init__(model_name_or_path=None)
        self.vectorizer = CountVectorizer(ngram_range=ngram_range, dtype=np.float32)
        self.is_fitted = False

    def fit(self, texts):
        self.vectorizer.fit(texts)
        self.is_fitted = True

    def encode(self, texts, **kwargs):
        if not self.is_fitted:
            self.fit(texts)
        return self.vectorizer.transform(texts).toarray()

    def transform(self, texts):
        return self.encode(texts)


class NgramAverageTransformer(SentenceTransformer):
    """Class that extracts embeddings by averaging over ngrams"""

    def __init__(
        self,
        checkpoint: str = "bert-base-uncased",
        ngrams: int = 3,
        layer: str = "last_hidden_state",
        all_ngrams: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Params
        ------
        checkpoint
            checkpoint name of the model
        ngrams: int
            What order of ngrams to use (1 for unigrams, 2 for bigrams, ...)
        layer: str
            which layer to extract embeddings from
        all_ngrams: bool
            whether to include all ngrams of lower order (should usually set to True)
        device: str
            device to run the model on ('cuda' for GPU, 'cpu' for CPU)
        """
        super().__init__(model_name_or_path=None)

        # save HF stuff
        # self.model_ = imodelsx.auglinear.embed.get_model(checkpoint)
        self.model_ = AutoModel.from_pretrained(checkpoint).to(device)

        # tokenizing the ngrams (word-based tokenization is more interpretable)
        self.tokenizer_ngrams_ = English().tokenizer

        # tokenizing for the embedding model
        self.tokenizer_embeddings_ = AutoTokenizer.from_pretrained(checkpoint)

        self.checkpoint = checkpoint
        self.ngrams = ngrams
        self.layer = layer
        self.all_ngrams = all_ngrams

    def fit(self, texts):
        # nothing happens here
        return self

    def encode(self, texts, batch_size=8, **kwargs):
        """
        Params
        ------
        batch_size: int
            batch size for simultaneously running ngrams (for a single example)
        """
        embs = []
        for x in texts:
            emb = imodelsx.auglinear.embed.embed_and_sum_function(
                x,
                model=self.model_,
                ngrams=self.ngrams,
                tokenizer_embeddings=self.tokenizer_embeddings_,
                tokenizer_ngrams=self.tokenizer_ngrams_,
                checkpoint=self.checkpoint,
                layer=self.layer,
                all_ngrams=self.all_ngrams,
                batch_size=batch_size,
            )
            embs.append(emb["embs"])
        return np.array(embs).squeeze()  # len(texts) x embedding_size

    def transform(self, texts):
        return self.encode(texts)

class QAembedder(SentenceTransformer):
    """Class that uses OpenAI's models to generate embeddings for question-answering tasks."""

    def __init__(
        self,
        questions_file,
        model_name="gpt-3.5-turbo",
        cache_file="QAembedderCache.json",
        system_prompt_file="QAembedder_system_prompt.txt",
        batch_size=1,
    ):
        super().__init__(model_name_or_path=None)
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.cache_file = cache_file
        self.questions_file = questions_file
        self.system_prompt_file = system_prompt_file
        self.batch_size = batch_size
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def fit(self, texts):
        # nothing happens here
        return self

    def encode(self, texts, **kwargs):
        responses_all = []
        with open(self.questions_file, "r") as f:
            questions = json.load(f)
        with open(self.system_prompt_file, "r") as f:
            system_prompt = f.read()
        for text in texts:
            if text in self.cache:
                responses_all.append(np.array(self.cache[text]))
                continue
            system_prompt = system_prompt + "\n" + text
            responses = []
            for i in range(0, len(questions), self.batch_size):
                batch_questions = questions[i : i + self.batch_size]
                user_prompt = [question["response"] for question in batch_questions]
                user_prompt = "Questions:\n" + str(user_prompt)
                message = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                response = self.openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=message,
                )
                # Assumes model output looks like a list. Convert the string that looks like a list into an actual list
                # TODO: use sglang, or guidance for structured generation.
                model_output_str = response["choices"][0]["message"]["content"]
                model_output = [
                    int(i) for i in model_output_str.strip("][").split(", ")
                ]

                if not all(val in [0, 1] for val in model_output):
                    print(f"Unexpected model output: {model_output}")
                    raise ValueError("Model output should only contain 0 or 1.")
                responses.extend(model_output)
            self.cache[text] = responses
            responses_all.append(np.array(responses))
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)
        return np.array(responses_all)

    def transform(self, texts):
        return self.encode(texts)

TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
    "SummEval",
]

TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)

vectorizers = {
    "TF-IDF-ngram-1": TfidfTransformer(ngram_range=(1, 1)),
    "TF-IDF-ngram-3": TfidfTransformer(ngram_range=(1, 3)),
    "TF-IDF-ngram-5": TfidfTransformer(ngram_range=(1, 5)),
    "BoW-ngram-1": BowTransformer(ngram_range=(1, 1)),
    "BoW-ngram-3": BowTransformer(ngram_range=(1, 3)),
    "BoW-ngram-5": BowTransformer(ngram_range=(1, 5)),
    "NgramEmbAverage": NgramAverageTransformer(),
}

if __name__ == "__main__":
    for model_name, model in vectorizers.items():
        for task in TASK_LIST:
            logger.info(f"Running task: {task} with model: {model_name}")
            eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
            evaluation = MTEB(tasks=[task], task_langs=["en"])
            evaluation.run(
                model,
                batch_size=2048,
                output_folder=f"results/{model_name}",
                eval_splits=eval_splits,
            )

    # TEST
    # texts = ["This is a test", "This is another test"]
    # model = BowTrNgramAverageTransformeransformer()
    # embs = model.encode(texts)
    # print(embs.shape)
